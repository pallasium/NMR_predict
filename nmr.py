"""NMR スペクトルの予測 (高速版; 1H, 13C, 31P, 19F, 11B と 2D: COSY, HSQC, HMQC, HMBC, TOCSY, NOESY, ROESY): RDKit 配座 -> xtb 最適化 -> ORCA GIAO (PBE/def2-SVP, 1 配座) + 経験式の J 結合 -> PNG。

使い方:
  python nmr.py "COc1ccccc1"                 # SMILES を直接指定
  python nmr.py Anisole                      # nmr.csv / nmr_local.csv の Compound_ID を指定
  python nmr.py "CCO" --name ethanol --solvent CDCl3 --nconf 5
  python nmr.py "Fc1ccccc1" --calc 19F,13C            # 計算するものを指定 (既定 1H)
  python nmr.py "CCO" --calc COSY,HSQC                  # 2D (必要な計算だけが走る)

出力: out/<name>_1H.png (J 結合は経験式), out/<name>_1H.csv (シフト), out/<name>_J.csv (J 結合)  (計算途中のファイルは work/<name>/ に残り、再実行時に再利用される)
設定: nmr_config.json (xtb, orca のパスなど。README 参照)
"""
import argparse, glob, hashlib, json, math, os, re, shutil, subprocess, sys
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Draw

HERE = os.path.dirname(os.path.abspath(__file__))
KCAL = 627.5095          # 1 Eh = 627.5095 kcal/mol
RT = 0.0019872 * 298.15  # kcal/mol
SOLVENT = {  # 重溶媒名 -> (xtb ALPB, ORCA CPCM)
    'cdcl3': ('chloroform', 'chloroform'), 'dmso': ('dmso', 'dmso'), 'dmso-d6': ('dmso', 'dmso'),
    'cd3od': ('methanol', 'methanol'), 'd2o': ('water', 'water'), 'acetone-d6': ('acetone', 'acetone'),
    'c6d6': ('benzene', 'benzene'),
}
NUC = {  # 核種 -> 元素, 1H に対する共鳴周波数の比, 基準物質 (SMILES), 表示する線幅 (Hz)
    '1H': {'el': 'H', 'ratio': 1.0, 'ref': 'C[Si](C)(C)C', 'fwhm': 1.2, 'refname': 'TMS'},
    '13C': {'el': 'C', 'ratio': 0.25145, 'ref': 'C[Si](C)(C)C', 'fwhm': 1.0, 'refname': 'TMS'},
    '31P': {'el': 'P', 'ratio': 0.40481, 'ref': 'OP(=O)(O)O', 'fwhm': 1.5, 'refname': 'H3PO4'},
    '19F': {'el': 'F', 'ratio': 0.94094, 'ref': 'FC(Cl)(Cl)Cl', 'fwhm': 2.0, 'refname': 'CFCl3'},
    '11B': {'el': 'B', 'ratio': 0.32084, 'ref': 'CC[O+](CC)[B-](F)(F)F', 'fwhm': 100.0, 'refname': 'BF3-OEt2'},  # 四極子核で線幅が広い
}
DEFAULTS = {'scale': {}, 'xtb': '', 'orca': '', 'nprocs': 'auto', 'maxcore_mb': 'auto', 'method': 'auto', 'basis': 'def2-SVP', 'orca_extra': 'auto', 'orca_conf': 1}


def total_ram_mb():
    """搭載メモリ (MB)。取れなければ None。"""
    try:
        if os.name == 'nt':
            import ctypes
            class MS(ctypes.Structure):
                _fields_ = [('l', ctypes.c_ulong), ('load', ctypes.c_ulong), ('total', ctypes.c_ulonglong),
                            ('avail', ctypes.c_ulonglong), ('a', ctypes.c_ulonglong), ('b', ctypes.c_ulonglong),
                            ('c', ctypes.c_ulonglong), ('d', ctypes.c_ulonglong), ('e', ctypes.c_ulonglong)]
            m = MS(); m.l = ctypes.sizeof(MS)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
            return m.avail / 2**20
        return os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_AVPHYS_PAGES') / 2**20
    except Exception:
        return None


def resolve_resources(cfg):
    """nprocs / maxcore_mb が 'auto' なら、論理コア数と空きメモリから決める。"""
    if str(cfg['nprocs']).lower() == 'auto':
        cfg['nprocs'] = os.cpu_count() or 1
    cfg['nprocs'] = int(cfg['nprocs'])
    if str(cfg['maxcore_mb']).lower() == 'auto':
        ram = total_ram_mb()
        # 空きメモリの 70% を全コアで割る (ORCA の %maxcore は 1 コアあたり。実使用は 2 倍程度まで増えうる)
        cfg['maxcore_mb'] = int(max(500, min(4000, 0.7 * ram / cfg['nprocs'] / 1.5))) if ram else 1000
    cfg['maxcore_mb'] = int(cfg['maxcore_mb'])
    return cfg


def resolve_method(cfg, need):
    """method が 'auto' なら、13C を含む計算 (13C, HSQC, HMBC) は PBE0、それ以外は PBE (RI 近似) にする。
    検証 (benchmark/): 13C の MAE は PBE 3.8 → PBE0 1.4 ppm、1H は両者でほぼ同じ (PBE が約 1.6〜1.9 倍速い)。
    nmr_config.json で method を指定したら、そのとおりにする。orca_extra が 'auto' なら、PBE は 'def2/J'、それ以外は空。"""
    cfg = dict(cfg)
    if str(cfg['method']).lower() == 'auto':
        cfg['method'] = 'PBE0' if '13C' in need else 'PBE'
    if str(cfg['orca_extra']).lower() == 'auto':
        cfg['orca_extra'] = 'def2/J' if cfg['method'].upper() == 'PBE' else ''
    return cfg


def load_config():
    cfg = dict(DEFAULTS)
    p = os.path.join(HERE, 'nmr_config.json')
    if os.path.exists(p):
        cfg.update(json.load(open(p, encoding='utf-8')))
    cfg['xtb'] = os.environ.get('XTB_PATH', cfg['xtb']) or shutil.which('xtb') or ''
    cfg['orca'] = os.environ.get('ORCA_PATH', cfg['orca']) or shutil.which('orca') or ''
    for k in ('xtb', 'orca'):
        if not cfg[k] or not os.path.exists(cfg[k]):
            sys.exit(f'エラー: {k} の場所が分かりません。nmr_config.json に "{k}" を書くか、環境変数 {k.upper()}_PATH を設定してください。')
    return cfg


def run(cmd, cwd, log, threads=None):
    env = dict(os.environ)
    if threads:
        env['OMP_NUM_THREADS'] = str(threads)
        env['MKL_NUM_THREADS'] = str(threads)
    with open(os.path.join(cwd, log), 'w', encoding='utf-8', errors='replace') as f:
        subprocess.run(cmd, cwd=cwd, stdout=f, stderr=subprocess.STDOUT, check=False, env=env)
    return open(os.path.join(cwd, log), encoding='utf-8', errors='replace').read()


def write_xyz(path, symbols, coords, comment=''):
    with open(path, 'w') as f:
        f.write(f'{len(symbols)}\n{comment}\n')
        for s, (x, y, z) in zip(symbols, coords):
            f.write(f'{s} {x:.8f} {y:.8f} {z:.8f}\n')


def read_xyz(path):
    lines = open(path).read().splitlines()
    n = int(lines[0])
    sym, xyz = [], []
    for l in lines[2:2 + n]:
        p = l.split()
        sym.append(p[0]); xyz.append([float(v) for v in p[1:4]])
    return sym, np.array(xyz)


def xtb_opt(cfg, d, symbols, coords, charge, alpb, threads=1):
    """xtb で最適化。(最適化後の座標, エネルギー Eh) を返す。結果があれば再利用。"""
    os.makedirs(d, exist_ok=True)
    opt = os.path.join(d, 'xtbopt.xyz')
    if not os.path.exists(opt):
        write_xyz(os.path.join(d, 'in.xyz'), symbols, coords)
        run([cfg['xtb'], 'in.xyz', '--gfn', '2', '--opt', 'normal', '--chrg', str(charge), '--alpb', alpb], d, 'xtb.log', threads)
    if not os.path.exists(opt):
        return None, math.inf
    txt = open(os.path.join(d, 'xtb.log'), encoding='utf-8', errors='replace').read()
    m = re.findall(r'TOTAL ENERGY\s+(-?\d+\.\d+)\s+Eh', txt)
    if not m:
        return None, math.inf
    return read_xyz(opt)[1], float(m[-1])


def orca_shieldings(cfg, d, symbols, coords, charge, cpcm):
    """ORCA GIAO。{原子 index: 等方遮蔽定数 ppm} を返す。結果があれば再利用。"""
    tag = re.sub(r'[^A-Za-z0-9_-]', '', f"{cfg['method']}_{cfg['basis']}_{cfg['orca_extra']}".replace('/', '-').replace(' ', '_'))
    d = os.path.join(d, 'orca_' + tag)                     # 手法ごとに別フォルダ (違う設定の結果を再利用しない)
    os.makedirs(d, exist_ok=True)
    out = os.path.join(d, 'orca.out')
    if not (os.path.exists(out) and 'CHEMICAL SHIELDING SUMMARY' in open(out, encoding='utf-8', errors='replace').read()):
        inp = [f"! {cfg['method']} {cfg['basis']} {cfg['orca_extra']} TightSCF NMR CPCM({cpcm})",
               f"%pal nprocs {cfg['nprocs']} end", f"%maxcore {cfg['maxcore_mb']}",
               f'* xyz {charge} 1']
        inp += [f'{s} {x:.8f} {y:.8f} {z:.8f}' for s, (x, y, z) in zip(symbols, coords)] + ['*']
        open(os.path.join(d, 'orca.inp'), 'w').write('\n'.join(inp) + '\n')
        run([cfg['orca'], 'orca.inp'], d, 'orca.out')
    txt = open(out, encoding='utf-8', errors='replace').read()
    i = txt.rfind('CHEMICAL SHIELDING SUMMARY')
    if i < 0:
        sys.exit(f'エラー: ORCA が失敗しました。{out} を確認してください。')
    res = {}
    for m in re.finditer(r'^\s*(\d+)\s+([A-Za-z]+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)', txt[i:], re.M):
        res[int(m.group(1))] = float(m.group(3))
    return res


def dihedral(p0, p1, p2, p3):
    """二面角 (度)。"""
    b0, b1, b2 = p0 - p1, p2 - p1, p3 - p2
    b1 = b1 / np.linalg.norm(b1)
    v = b0 - np.dot(b0, b1) * b1
    w = b2 - np.dot(b2, b1) * b1
    return math.degrees(math.atan2(np.dot(np.cross(b1, v), w), np.dot(v, w)))


def empirical_couplings(mol, xyz):
    """構造 (xtb 最適化後の座標) から H-H の J (Hz) を経験式で見積もる。{(i, j): J} (i<j)。
    3J(sp3-sp3): Haasnoot-Altona 型 J = 13.7 cos^2(phi) - 0.73 cos(phi)  (置換基の補正なし)
    3J(アルケン): cis 10.5 / trans 16.5 Hz (二面角で判定)。ベンゼン環: オルト 7.6, メタ 1.5, パラ 0.6 Hz。
    5 員環の芳香環の 3J は 3.5 Hz。アリル型 4J 1.5 Hz。それ以外は 0 とする。"""
    dm = Chem.GetDistanceMatrix(mol)
    ri = mol.GetRingInfo()
    hs = [a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() == 'H']
    heavy = lambda h: mol.GetAtomWithIdx(h).GetNeighbors()[0].GetIdx()
    sp2 = lambda i: mol.GetAtomWithIdx(i).GetHybridization() in (Chem.HybridizationType.SP2, Chem.HybridizationType.SP)
    sp3 = lambda i: mol.GetAtomWithIdx(i).GetHybridization() == Chem.HybridizationType.SP3
    arom = lambda i: mol.GetAtomWithIdx(i).GetIsAromatic()
    def double_bond(i, j):
        b = mol.GetBondBetweenAtoms(i, j)
        return b is not None and b.GetBondType() == Chem.BondType.DOUBLE
    def ring_size(i, j):
        rs = [len(r) for r in ri.AtomRings() if i in r and j in r]
        return min(rs) if rs else 0
    def is_carbonyl_c(i):
        return any(n.GetSymbol() == 'O' and double_bond(i, n.GetIdx()) for n in mol.GetAtomWithIdx(i).GetNeighbors())
    J = {}
    for x in range(len(hs)):
        for y in range(x + 1, len(hs)):
            i, j = hs[x], hs[y]
            ci, cj = heavy(i), heavy(j)
            if ci == cj or mol.GetAtomWithIdx(ci).GetSymbol() != 'C' or mol.GetAtomWithIdx(cj).GetSymbol() != 'C':
                continue
            d = int(dm[i][j])
            v = 0.0
            if d == 3:                                              # H-C-C-H
                phi = dihedral(xyz[i], xyz[ci], xyz[cj], xyz[j])
                c = math.cos(math.radians(phi))
                if arom(ci) and arom(cj):
                    v = 3.5 if ring_size(ci, cj) == 5 else 7.6
                elif double_bond(ci, cj):
                    v = 10.5 if abs(phi) < 90 else 16.5
                elif sp3(ci) and sp3(cj):
                    v = max(0.0, 13.7 * c * c - 0.73 * c)
                elif (sp3(ci) and sp2(cj)) or (sp2(ci) and sp3(cj)):
                    v = 2.5 if (is_carbonyl_c(ci) or is_carbonyl_c(cj)) else 6.5
                elif sp2(ci) and sp2(cj):
                    v = 10.0                                        # 共役ジエンなど
            elif d == 4:                                            # 4 結合
                if arom(ci) and arom(cj) and dm[ci][cj] == 2 and ring_size(ci, cj) == 6:
                    v = 1.5                                         # メタ
                elif sp2(ci) != sp2(cj):                            # H-C=C-C-H (アリル型)
                    e, f = (ci, cj) if sp2(ci) else (cj, ci)
                    if any(double_bond(e, n.GetIdx()) and dm[n.GetIdx()][f] == 1 for n in mol.GetAtomWithIdx(e).GetNeighbors()):
                        v = 1.5
            elif d == 5:                                            # パラ
                if arom(ci) and arom(cj) and dm[ci][cj] == 3 and ring_size(ci, cj) == 6:
                    v = 0.6
            if v:
                J[(i, j)] = v
    return J


def simulate_spin_system(nu, J, max_spins=12):
    """核スピン 1/2 の系の厳密なスペクトル (2 次の効果を含む)。nu: 共鳴周波数 Hz, J: 結合定数行列 Hz。
    戻り値: (線の周波数 Hz, 強度)。強度の総和は核の数に規格化する。"""
    n = len(nu)
    freqs, ints = [], []
    sectors = {}
    for k in range(n + 1):                                    # k = 上向きスピンの数
        states = [sum(1 << b for b in c) for c in __import__('itertools').combinations(range(n), k)]
        idx = {st: i for i, st in enumerate(states)}
        H = np.zeros((len(states), len(states)))
        for st, a in idx.items():
            sz = [0.5 if (st >> b) & 1 else -0.5 for b in range(n)]
            H[a, a] = sum(nu[b] * sz[b] for b in range(n))
            for b in range(n):
                for c in range(b + 1, n):
                    if J[b, c] == 0.0:
                        continue
                    H[a, a] += J[b, c] * sz[b] * sz[c]
                    if sz[b] != sz[c]:                        # I+I- + I-I+ の項
                        H[idx[st ^ (1 << b) ^ (1 << c)], a] += 0.5 * J[b, c]
        E, V = np.linalg.eigh(H)
        sectors[k] = (states, idx, E, V)
    for k in range(1, n + 1):                                 # k -> k-1 の遷移 (F-)
        st_k, _, E_k, V_k = sectors[k]
        _, idx_l, E_l, V_l = sectors[k - 1]
        T = np.zeros((len(idx_l), len(st_k)))
        for a, st in enumerate(st_k):
            for b in range(n):
                if (st >> b) & 1:
                    T[idx_l[st ^ (1 << b)], a] = 1.0
        M = V_l.T @ T @ V_k                                   # [下, 上]
        I = M ** 2
        sel = I > 1e-6 * I.max()
        fr = (E_k[None, :] - E_l[:, None])[sel]
        freqs.append(fr); ints.append(I[sel])
    f = np.concatenate(freqs); w = np.concatenate(ints)
    return f, w * n / w.sum()


def split_components(hs, J, shift, max_spins):
    """結合のつながりごとに分ける。max_spins を超える系は、化学シフトが近い H 同士の結合から順に切って分割する
    (近い H 同士の結合は、切っても見た目への影響が小さい。離れた H 同士の分裂は残る)。戻り値: (成分のリスト, 残った J, 切った本数)"""
    J = dict(J)
    cut = 0
    while True:
        adj = {h: set() for h in hs}
        for (a, b) in J:
            adj[a].add(b); adj[b].add(a)
        seen, comps = set(), []
        for h in hs:
            if h in seen:
                continue
            comp, stack = [], [h]
            while stack:
                x = stack.pop()
                if x in seen:
                    continue
                seen.add(x); comp.append(x); stack.extend(adj[x])
            comps.append(sorted(comp))
        big = max(comps, key=len)
        if len(big) <= max_spins:
            return comps, J, cut
        inside = [e for e in J if e[0] in big and e[1] in big]
        e = min(inside, key=lambda e: (abs(shift[e[0]] - shift[e[1]]), abs(J[e])))
        del J[e]; cut += 1


def pair_couplings(mol, peaks, jmat, jmin=0.5):
    """H-H の J を、CH3 / CH2 の回転平均をして整理する。戻り値: (shift, group, 交換性 H の集合, J)。
    shift/group: {H の原子 index: ppm / peaks の番号}, J: {(i, j): Hz} (|J| >= jmin のものだけ)。"""
    ranks = list(Chem.CanonicalRankAtoms(mol, breakTies=False))
    shift, group, exch = {}, {}, set()
    for gi, (d, n, idxs, ex) in enumerate(peaks):
        for i in idxs:
            shift[i] = d; group[i] = gi
            if ex:
                exch.add(i)
    hs = sorted(shift)
    def same_set(i):                      # 同じ炭素上の等価な H (CH3, CH2): 回転で平均される
        c = mol.GetAtomWithIdx(i).GetNeighbors()[0].GetIdx()
        return [h.GetIdx() for h in mol.GetAtomWithIdx(c).GetNeighbors()
                if h.GetSymbol() == 'H' and ranks[h.GetIdx()] == ranks[i]]
    def Jraw(a, b):
        return jmat.get((min(a, b), max(a, b)), 0.0)
    J = {}
    for a in hs:
        for b in hs:
            if a < b and a not in exch and b not in exch:
                Sa, Sb = same_set(a), same_set(b)
                if b in Sa:
                    continue                                   # 同じ CH3/CH2 内は等価で無関係
                v = float(np.mean([Jraw(x, y) for x in Sa for y in Sb]))
                if abs(v) >= jmin:
                    J[(a, b)] = v
    return shift, group, exch, J


def build_spectrum(mol, peaks, jmat, mhz, jmin=0.5, max_spins=12):
    """peaks: peaks_from の戻り値。jmat: {(i,j): J} (配座平均済み)。
    戻り値: (線の位置 ppm, 強度) と、グループ間の J の一覧。"""
    shift, group, exch, J = pair_couplings(mol, peaks, jmat, jmin)
    hs = sorted(shift)
    # 結合のつながり (連結成分) ごとにシミュレーション
    comps, Jc, cut = split_components(hs, J, shift, max_spins)
    if cut:
        print(f'  注意: 結合した H が多い系なので、化学シフトの近い H 同士の結合 {cut} 本を無視して分割しました (近似)')
    lines, ws = [], []
    for comp in comps:
        nu = np.array([shift[i] * mhz for i in comp])
        Jm = np.zeros((len(comp), len(comp)))
        for ia, a in enumerate(comp):
            for ib, b in enumerate(comp):
                if (a, b) in Jc:
                    Jm[ia, ib] = Jm[ib, ia] = Jc[(a, b)]
        if len(comp) == 1 or not Jm.any():
            for i in comp:
                lines.append(np.array([shift[i]])); ws.append(np.array([1.0]))
            continue
        f, w = simulate_spin_system(nu, Jm)
        lines.append(f / mhz); ws.append(w)
    for i in exch:                                             # 交換性 H は 1 本の線
        lines.append(np.array([shift[i]])); ws.append(np.array([1.0]))
    tab = {}
    for (a, b), v in J.items():
        ga, gb = group[a], group[b]
        tab.setdefault((min(ga, gb), max(ga, gb)), []).append(v)
    gtab = [(peaks[ga][0], peaks[gb][0], float(np.mean(v)), len(v)) for (ga, gb), v in sorted(tab.items())]
    return np.concatenate(lines), np.concatenate(ws), gtab


MULT_NAME = {1: 'd', 2: 't', 3: 'q', 4: 'quint', 5: 'sext', 6: 'sept'}


def multiplets(mol, peaks, jmat, mhz, jmin=0.5, jres=1.0, jtol=1.0):
    """各 1H ピークの多重度と J (Hz) を、一次の分裂則で決める。戻り値: peaks と同じ順の [(多重度, [J (Hz, 大きい順)])]。
    相手のグループごとに (等価な H の数, J) を数え、J が jtol 以内のものは 1 組にまとめる (n+1 則)。
    次の場合は 'm' (多重線) とする: 分裂の組が 4 つ以上 / 線が 16 本を超える / 化学シフト差 (Hz) が J の 4 倍未満 (2 次の効果) /
    隣のピークと重なる。交換性 H は 'br s'。"""
    shift, group, exch, J = pair_couplings(mol, peaks, jmat, jmin)
    adj = {}
    for (a, b), v in J.items():
        adj.setdefault(a, []).append((b, v)); adj.setdefault(b, []).append((a, v))
    res, span = [], []
    for gi, (d, n, idxs, ex) in enumerate(peaks):
        if ex:
            res.append(('br s', [])); span.append(2.0)
            continue
        per = {}                                                  # 相手のグループ -> この群の各 H から見た J の一覧
        for a in idxs:
            by = {}
            for b, v in adj.get(a, []):
                if group[b] != gi:
                    by.setdefault(group[b], []).append(abs(v))
            for g2, vs in by.items():
                per.setdefault(g2, []).append(vs)
        sets, strong = [], False
        for g2, lst in per.items():                               # 同じ相手グループでも、J が違う H (オルト / パラ など) は別の組にする
            cnt = int(sum(len(v) for v in lst) / len(idxs) + 0.5)
            if cnt < 1:
                continue
            cols = np.mean([sorted(v, reverse=True)[:cnt] + [0.0] * (cnt - len(v)) for v in lst], axis=0)
            for jm in cols:
                if jm >= jres:
                    sets.append((float(jm), 1))
                    if abs(peaks[g2][0] - d) * mhz < 4 * jm:
                        strong = True
        sets.sort(reverse=True)
        cl = []                                                   # [J の重み付き和, 本数, 数]
        for jm, cnt in sets:
            if cl and abs(jm - cl[-1][0] / cl[-1][1]) <= jtol:
                cl[-1][0] += jm * cnt; cl[-1][1] += cnt
            else:
                cl.append([jm * cnt, cnt])
        cl = [(s / c, c) for s, c in cl]
        nlines = int(np.prod([c + 1 for _, c in cl])) if cl else 1
        if not cl:
            lab = 's'
        elif strong or len(cl) > 3 or nlines > 16 or any(c > 6 for _, c in cl) or (len(cl) > 1 and any(c >= 4 for _, c in cl)):
            lab = 'm'
        else:
            lab = ''.join(MULT_NAME[c] for _, c in cl)
        res.append((lab, [j for j, _ in cl] if lab not in ('s', 'm') else []))
        span.append(max(2.0, sum(j * c for j, c in cl)))
    for k, (d, _, _, ex) in enumerate(peaks):                     # 隣のピークと重なる多重線は m
        if res[k][0] in ('s', 'br s', 'm'):
            continue
        for k2, (d2, _, _, ex2) in enumerate(peaks):
            if k2 != k and not ex2 and abs(d - d2) * mhz < (span[k] + span[k2]) / 2:
                res[k] = ('m', []); break
    return res


def mult_text(lab, js, short=False):
    """'d, J = 8.1 Hz' / 'dd, J = 8.1, 2.0 Hz' (short: 'd 8.1' / 'dd 8.1/2.0')。"""
    if not js:
        return lab
    if short:
        return lab + ' ' + '/'.join(f'{j:.1f}' for j in js)
    return lab + ', J = ' + ', '.join(f'{j:.1f}' for j in js) + ' Hz'


def report_1H(peaks, mults, mhz):
    """論文の形式の 1 行: 1H NMR (400 MHz): δ 7.45 (d, J = 8.1 Hz, 2H), ..."""
    parts = [f'{d:.2f} ({mult_text(*m)}, {n}H)' for (d, n, _, _), m in zip(peaks, mults)]
    return f'1H NMR ({mhz:.0f} MHz): δ ' + ', '.join(parts)


def conformers(mol, nconf):
    """RDKit で配座生成 -> MMFF -> エネルギーの低い順に nconf 個。"""
    ps = AllChem.ETKDGv3(); ps.randomSeed = 42; ps.pruneRmsThresh = 0.5
    ids = list(AllChem.EmbedMultipleConfs(mol, 30, ps))
    if not ids:
        sys.exit('エラー: 3D 構造を作れませんでした。SMILES を確認してください。')
    res = AllChem.MMFFOptimizeMoleculeConfs(mol, maxIters=2000)
    order = sorted(range(len(ids)), key=lambda i: res[i][1])
    return [ids[i] for i in order[:nconf]]


def compute_shieldings(cfg, mol, workdir, alpb, cpcm, nconf, max_orca=1, want_j=False, conf_out=None):
    """ボルツマン平均した遮蔽定数を返す。{H の原子 index: σ}"""
    charge = Chem.GetFormalCharge(mol)
    if sum(a.GetNumRadicalElectrons() for a in mol.GetAtoms()):
        sys.exit('エラー: ラジカルには未対応です。')
    symbols = [a.GetSymbol() for a in mol.GetAtoms()]
    from concurrent.futures import ThreadPoolExecutor
    cids = conformers(mol, nconf)
    jobs = max(1, min(len(cids), cfg['nprocs']))      # 配座は並列に、1 ジョブあたりのスレッド数は残りのコアで割る
    threads = max(1, cfg['nprocs'] // jobs)
    def job(k):
        xyz = mol.GetConformer(cids[k]).GetPositions()
        c, e = xtb_opt(cfg, os.path.join(workdir, f'conf{k}'), symbols, xyz, charge, alpb, threads)
        return e, k, c
    with ThreadPoolExecutor(jobs) as ex:
        confs = [r for r in ex.map(job, range(len(cids))) if r[2] is not None]
    if not confs:
        sys.exit('エラー: xtb の最適化がすべて失敗しました。')
    confs.sort(key=lambda t: t[0])
    e0 = confs[0][0]
    w = np.array([math.exp(-(e - e0) * KCAL / RT) for e, _, _ in confs]); w /= w.sum()
    w_all = w.copy()
    if conf_out is not None:                            # NOESY 用: xtb の全配座の (重み, 座標)
        conf_out.extend((float(wi), c) for wi, (_, _, c) in zip(w_all, confs))
    keep = [i for i in range(len(confs)) if w[i] >= 0.05][:max_orca]
    w = np.array([w[i] for i in keep]); w /= w.sum()
    print(f'  配座 {len(confs)} 個 (xtb) -> ORCA は {len(keep)} 個: 重み ' + ', '.join(f'{x:.2f}' for x in w))
    acc, jacc = {}, {}
    if want_j:                                          # J は安いので xtb の全配座でボルツマン平均
        for wi, (_, _, c) in zip(w_all, confs):
            for pair, v in empirical_couplings(mol, c).items():
                jacc[pair] = jacc.get(pair, 0.0) + wi * v
    for wi, i in zip(w, keep):
        _, k, c = confs[i]
        print(f'  ORCA GIAO conf{k} ...', flush=True)
        sh = orca_shieldings(cfg, os.path.join(workdir, f'conf{k}'), symbols, c, charge, cpcm)
        for idx, s in sh.items():
            acc[idx] = acc.get(idx, 0.0) + wi * s
    return (acc, jacc) if want_j else acc


def compute_all(cfg, mol, workdir, alpb, cpcm, nconf, max_orca=1, want_j=False, conf_out=None):
    """分子が複数の断片 (塩: カチオン + アニオン など) に分かれるときは、断片ごとに別々に計算して結果をまとめる。
    イオン対のまま DFT (PBE) で計算すると、アニオンからカチオンへの人工的な電荷移動で、遮蔽定数が大きく狂う
    (13C で約 50 ppm) ため。イオン対の相互作用 (水素結合など) は考えない。"""
    maps = []
    frags = Chem.GetMolFrags(mol, asMols=True, fragsMolAtomMapping=maps)
    if len(frags) == 1:
        return compute_shieldings(cfg, mol, workdir, alpb, cpcm, nconf, max_orca, want_j, conf_out)
    print(f'  {len(frags)} 個の断片 (イオンなど) に分けて、別々に計算します')
    sig, J, fconfs = {}, {}, []
    for k, (fm, mp) in enumerate(zip(frags, maps)):
        co = [] if conf_out is not None else None
        r = compute_shieldings(cfg, fm, f'{workdir}_frag{k}', alpb, cpcm, nconf, max_orca, want_j, co)
        s_, j_ = r if want_j else (r, {})
        for i, v in s_.items():
            sig[mp[i]] = v
        for (a, b), v in j_.items():
            J[(min(mp[a], mp[b]), max(mp[a], mp[b]))] = v
        fconfs.append((mp, co))
    if conf_out is not None:                                   # NOESY 用: 断片どうしは遠く離して、断片間の NOE が出ないようにする
        n = mol.GetNumAtoms()
        base_mp, base_co = max(fconfs, key=lambda t: len(t[0]))
        for w, xyz in base_co:
            full = np.zeros((n, 3))
            for kk, (mp, co) in enumerate(fconfs):
                full[list(mp)] = (xyz if mp is base_mp else co[0][1]) + 1000.0 * kk
            conf_out.append((w, full))
    return (sig, J) if want_j else sig


def reference_sigma(cfg, nuc, alpb, cpcm):
    """基準物質 (1H, 13C: TMS / 31P: H3PO4 / 19F: CFCl3 / 11B: BF3-OEt2) の遮蔽定数 (その元素の原子の平均)。条件ごとに cache に保存。"""
    spec = NUC[nuc]
    key = f"{cfg['method']}_{cfg['basis']}_{cfg['orca_extra']}_{cpcm}".replace('/', '-').replace(' ', '')
    p = os.path.join(HERE, 'cache', f'tms_{key}.json' if nuc == '1H' else f'ref_{nuc}_{key}.json')
    if os.path.exists(p):
        return json.load(open(p))['sigma']
    print(f'基準物質 {spec["refname"]} を計算します。初回のみです ...')
    mol = Chem.AddHs(Chem.MolFromSmiles(spec['ref']))
    wd = os.path.join(HERE, 'work', f'_TMS_{key}' if spec['ref'] == 'C[Si](C)(C)C' else f'_REF_{spec["refname"]}_{key}')
    sh = compute_shieldings(cfg, mol, wd, alpb, cpcm, nconf=1)
    vals = [v for i, v in sh.items() if mol.GetAtomWithIdx(i).GetSymbol() == spec['el']]
    sigma = float(np.mean(vals))
    os.makedirs(os.path.dirname(p), exist_ok=True)
    json.dump({'sigma': sigma, 'n_atoms': len(vals), 'ref': spec['refname']}, open(p, 'w'))
    return sigma


TWOD = ['COSY', 'HSQC', 'HMQC', 'HMBC', 'TOCSY', 'NOESY', 'ROESY']


def parse_calc(text):
    """'1H,13C,COSY' のような指定を、計算するもののリストにする。all = 全核種, 2D = 全部の 2 次元。空なら ['1H']。
    戻り値は 1D (NUC の順) の後に 2D (TWOD の順)。未対応の名前があれば ValueError。"""
    items = [t.strip() for t in text.replace('、', ',').split(',') if t.strip()]
    if not items:
        return ['1H']
    want = set()
    for t in items:
        u = t.upper()
        if u == 'ALL':
            want |= set(NUC)
        elif u == '2D':
            want |= set(TWOD)
        elif u in NUC or u in TWOD:
            want.add(u)
        else:
            raise ValueError(t)
    return [n for n in NUC if n in want] + [n for n in TWOD if n in want]


def atom_shifts(peaks):
    return {i: d for d, n, idxs, ex in peaks for i in idxs}


def group_J_table(J, group):
    """グループ間の J (平均): {(ga, gb): Hz} (ga < gb)。"""
    tab = {}
    for (a, b), v in J.items():
        ga, gb = group[a], group[b]
        if ga != gb:
            tab.setdefault((min(ga, gb), max(ga, gb)), []).append(v)
    return {k: float(np.mean(v)) for k, v in tab.items()}


def _diag(peaksH):
    return [(d, d, float(n), 'diag', f'{d:.2f} ({n}H)') for d, n, idxs, ex in peaksH if not ex]


def cosy_points(peaksH, J, group, jmin):
    """COSY: 対角 + J が jmin (Hz) 以上の H グループの組の相関ピーク。"""
    pts = _diag(peaksH)
    for (ga, gb), v in group_J_table(J, group).items():
        if abs(v) < jmin:
            continue
        (da, na), (db, nb) = peaksH[ga][:2], peaksH[gb][:2]
        w = 0.6 * min(1.0, abs(v) / 8.0) * math.sqrt(na * nb)
        note = f'{da:.2f}-{db:.2f} ppm, J={v:.1f} Hz'
        pts += [(da, db, w, 'cross', note), (db, da, w, 'cross', note)]
    return pts


def tocsy_points(peaksH, J, group, jmin):
    """TOCSY: 結合でつながった H グループ (スピン系) の全部の組。つながりが遠いほど弱い。"""
    tab = group_J_table(J, group)
    adj = {}
    for (a, b), v in tab.items():
        if abs(v) >= jmin:
            adj.setdefault(a, set()).add(b); adj.setdefault(b, set()).add(a)
    pts = _diag(peaksH)
    for a in adj:
        dist, queue = {a: 0}, [a]
        while queue:
            x = queue.pop(0)
            for y in adj[x]:
                if y not in dist:
                    dist[y] = dist[x] + 1; queue.append(y)
        for b, hops in dist.items():
            if b != a:
                (da, na), (db, nb) = peaksH[a][:2], peaksH[b][:2]
                w = 0.6 * {1: 1.0, 2: 0.6, 3: 0.4}.get(hops, 0.25) * math.sqrt(na * nb)
                pts.append((da, db, w, 'cross', f'{da:.2f}-{db:.2f} ppm, {hops} 段'))
    return pts


def hsqc_points(mol, peaksH, peaksC, edited=True):
    """HSQC / HMQC の C-H 直結 (1 結合)。HSQC (edited=True) は CH, CH3 を赤 (+)、CH2 を青 (-) の編集型で描く。
    HMQC (edited=False) は多重度の編集がないので、すべて同じ種類 ('hmqc')。x = 1H, y = 13C。"""
    sh, sc = atom_shifts(peaksH), atom_shifts(peaksC)
    acc = {}
    for at in mol.GetAtoms():
        if at.GetSymbol() != 'C' or at.GetIdx() not in sc:
            continue
        hs = [n.GetIdx() for n in at.GetNeighbors() if n.GetSymbol() == 'H' and n.GetIdx() in sh]
        for h in hs:
            kind = ('neg' if len(hs) == 2 else 'pos') if edited else 'hmqc'
            key = (round(sh[h], 3), round(sc[at.GetIdx()], 2), kind)
            acc[key] = acc.get(key, 0.0) + 1.0 / len(hs)
    return [(x, y, w, k, f'1H {x:.2f} / 13C {y:.1f}') for (x, y, k), w in acc.items()]


def hmbc_points(mol, peaksH, peaksC):
    """HMBC: 2 結合 (弱) と 3 結合 (強) 離れた C-H の相関。x = 1H, y = 13C。"""
    sh, sc = atom_shifts(peaksH), atom_shifts(peaksC)
    ex = {i for d, n, idxs, e in peaksH if e for i in idxs}
    dm = Chem.GetDistanceMatrix(mol)
    acc = {}
    for c in sc:
        for h in sh:
            if h in ex:
                continue
            d = int(dm[c][h])
            if d in (2, 3):
                key = (round(sh[h], 3), round(sc[c], 2))
                acc[key] = acc.get(key, 0.0) + (0.5 if d == 2 else 1.0)
    return [(x, y, min(w, 2.0), 'hmbc', f'1H {x:.2f} / 13C {y:.1f}') for (x, y), w in acc.items()]


def noesy_points(peaksH, confs, rmax=5.0, cls='cross'):
    """NOESY / ROESY: 空間的に近い H グループの組。強度は <r^-6> のボルツマン平均 (xtb の配座) の和。
    ROESY (cls='roe') は、交差ピークが分子量によらず対角と逆の位相 (NOE の符号反転 = ゼロ交差がない)。"""
    ex = {i for d, n, idxs, e in peaksH if e for i in idxs}
    hs = [i for d, n, idxs, e in peaksH if not e for i in idxs]
    pos = {h: k for k, h in enumerate(hs)}
    R6 = np.zeros((len(hs), len(hs)))
    for w, xyz in confs:
        X = np.asarray(xyz)[hs]
        D = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=2)
        np.fill_diagonal(D, np.inf)
        R6 += w * D ** -6.0
    pts = _diag(peaksH)
    thr = (2.5 / rmax) ** 6
    for ga, (da, na, ia, ea) in enumerate(peaksH):
        for gb, (db, nb, ib, eb) in enumerate(peaksH):
            if gb <= ga or ea or eb:
                continue
            I = sum(R6[pos[a], pos[b]] for a in ia for b in ib) * 2.5 ** 6
            if I < thr:
                continue
            w = 0.6 * min(1.0, math.sqrt(I)) * math.sqrt(na * nb)
            note = f'{da:.2f}-{db:.2f} ppm, 実効距離 約 {2.5 * I ** (-1 / 6):.1f} A'
            pts += [(da, db, w, cls, note), (db, da, w, cls, note)]
    return pts


def plot_2d(kind, pts, nx, ny, path, title, mol0, tx, ty):
    """2D スペクトル。pts: [(x, y, 強度, 種類, 注記)]。nx, ny: 横軸・縦軸の核種。tx, ty: 上と左に出す 1D の [(shift, 重み)]。"""
    import matplotlib
    matplotlib.use('Agg')
    matplotlib.rcParams['font.family'] = ['Yu Gothic', 'Meiryo', 'MS Gothic', 'DejaVu Sans']
    import matplotlib.pyplot as plt
    pad = {'1H': 0.5, '13C': 10.0}
    xs = [p[0] for p in pts] + [t[0] for t in tx]
    ys = [p[1] for p in pts] + [t[0] for t in ty]
    xlo, xhi = min(xs) - pad[nx], max(xs) + pad[nx]
    ylo, yhi = min(ys) - pad[ny], max(ys) + pad[ny]
    if nx == ny:
        xlo = ylo = min(xlo, ylo); xhi = yhi = max(xhi, yhi)
    gx, gy = np.linspace(xlo, xhi, 700), np.linspace(ylo, yhi, 700)
    sx = max(0.012 if nx == '1H' else 0.4, (xhi - xlo) / 170)
    sy = max(0.012 if ny == '1H' else 0.4, (yhi - ylo) / 170)
    colors = {'diag': '#555555', 'cross': 'tab:red', 'pos': 'tab:red', 'neg': 'tab:blue', 'hmbc': 'tab:green', 'hmqc': 'tab:purple', 'roe': 'tab:blue'}
    Z = {}
    for x, y, w, cls, _ in pts:
        z = Z.setdefault(cls, np.zeros((len(gy), len(gx))))
        z += w * np.outer(np.exp(-(gy - y) ** 2 / (2 * sy ** 2)), np.exp(-(gx - x) ** 2 / (2 * sx ** 2)))
    zmax = max(z.max() for z in Z.values())
    fig = plt.figure(figsize=(8, 8))
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 5], height_ratios=[1, 5], wspace=0.03, hspace=0.03)
    ax = fig.add_subplot(gs[1, 1]); axt = fig.add_subplot(gs[0, 1], sharex=ax)
    axl = fig.add_subplot(gs[1, 0], sharey=ax); axc = fig.add_subplot(gs[0, 0])
    for cls, z in Z.items():
        levels = [zmax * f for f in (0.02, 0.05, 0.12, 0.3, 0.6) if zmax * f < z.max()]
        if levels:
            ax.contour(gx, gy, z, levels=levels, colors=colors[cls], linewidths=0.8)
    for cls in Z:
        sel = [(p[0], p[1]) for p in pts if p[3] == cls]
        ax.scatter([a for a, _ in sel], [b for _, b in sel], marker='+', s=14, color=colors[cls], linewidths=0.6)
    if nx == ny:
        ax.plot([xlo, xhi], [xlo, xhi], ':', color='gray', lw=0.5)
    ax.set_xlim(xhi, xlo); ax.set_ylim(ylo, yhi)                 # 横軸は左が高磁場側 (大きい ppm)、縦軸は上が大きい ppm
    ax.set_xlabel(f'δ {nx} (ppm)'); ax.set_ylabel(f'δ {ny} (ppm)')
    ax.yaxis.set_label_position('right'); ax.yaxis.tick_right()
    axt.plot(gx, sum((w * np.exp(-(gx - d) ** 2 / (2 * sx ** 2)) for d, w in tx), np.zeros_like(gx)), color='black', lw=0.8)
    axl.plot(sum((w * np.exp(-(gy - d) ** 2 / (2 * sy ** 2)) for d, w in ty), np.zeros_like(gy)), gy, color='black', lw=0.8)
    axl.invert_xaxis()
    axt.tick_params(bottom=False, labelbottom=False, left=False, labelleft=False)    # 目盛りは共有軸なので set_xticks は使わない
    axl.tick_params(bottom=False, labelbottom=False, left=False, labelleft=False)
    for a_ in (axt, axl):
        for sp in a_.spines.values():
            sp.set_visible(False)
    axc.imshow(Draw.MolToImage(Chem.RemoveHs(mol0), size=(220, 160))); axc.axis('off')
    fig.suptitle(f'{title}   ({kind}, predicted)', fontsize=10)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def exchangeable(mol, idx):
    """O-H / N-H / S-H の H (溶媒・濃度で大きく動くので参考値扱い)。"""
    return mol.GetAtomWithIdx(idx).GetNeighbors()[0].GetSymbol() in ('O', 'N', 'S')


def peaks_from(mol, sigma, sigma_ref, el='H', scale=None):
    """対称等価な原子をまとめて [(shift ppm, 原子数, [原子 index], 交換性 H か)]。
    shift = slope * (σ基準 - σ) + intercept (scale = [slope, intercept]。省略時は 1, 0)。"""
    sl, ic = scale if scale else (1.0, 0.0)
    ranks = list(Chem.CanonicalRankAtoms(mol, breakTies=False))
    groups = {}
    for idx in sigma:
        groups.setdefault(ranks[idx], []).append(idx)
    peaks = []
    for idxs in groups.values():
        delta = sl * float(np.mean([sigma_ref - sigma[i] for i in idxs])) + ic
        peaks.append((delta, len(idxs), sorted(idxs), exchangeable(mol, idxs[0]) if el == 'H' else False))
    return sorted(peaks, key=lambda p: -p[0])


def plot_nuc(mol0, peaks, path, title, nuc, mhz):
    """異核種 (13C, 31P, 19F, 11B) のスペクトル。プロトン脱カップリングの 1 本線 (面積 ∝ 原子数)。"""
    import matplotlib
    matplotlib.use('Agg')
    matplotlib.rcParams['font.family'] = ['Yu Gothic', 'Meiryo', 'MS Gothic', 'DejaVu Sans']
    import matplotlib.pyplot as plt
    spec = NUC[nuc]
    pos = [p[0] for p in peaks]
    margin = max({'13C': 10, '31P': 20, '19F': 20, '11B': 10}.get(nuc, 10), 0.08 * (max(pos) - min(pos)))
    lo, hi = min(pos) - margin, max(pos) + margin
    x = np.linspace(lo, hi, 20000)
    y = lorentz_sum(x, (np.array(pos), np.array([float(p[1]) for p in peaks])), spec['fwhm'] / (mhz * spec['ratio']))
    fig, ax = plt.subplots(figsize=(10, 4.6))
    ax.plot(x, y, color='black', lw=0.9)
    ax.set_xlim(hi, lo); ax.set_ylim(0, y.max() * 2.3)
    ax.set_yticks([]); ax.set_xlabel(f'δ {nuc} (ppm)')
    for sp in ('top', 'right', 'left'):
        ax.spines[sp].set_visible(False)
    for d, n, _, _ in peaks:
        ax.annotate(f'{d:.1f} ({n})', (d, 0), xytext=(d, y[np.argmin(np.abs(x - d))] + 0.03 * y.max()),
                    rotation=90, ha='center', va='bottom', fontsize=8)
    ax.set_title(f'{title}   (predicted {nuc}, {mhz * spec["ratio"]:.0f} MHz, singlets, ref {spec["refname"]})', fontsize=10)
    img = Draw.MolToImage(Chem.RemoveHs(mol0), size=(260, 190))
    ins = fig.add_axes([0.70, 0.62, 0.27, 0.32]); ins.imshow(img); ins.axis('off')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def lorentz_sum_exact(x, sticks, fwhm_ppm):
    """線を 1 本ずつローレンツ型で足す (厳密。線の数 × 点の数だけかかる)。検証用。"""
    g = fwhm_ppm / 2
    y = np.zeros_like(x)
    for c0 in range(0, len(sticks[0]), 500):                       # 線の数が多いので分割して足す
        fx, fw = sticks[0][c0:c0 + 500], sticks[1][c0:c0 + 500]
        y += (fw[:, None] * g * g / ((x[None, :] - fx[:, None]) ** 2 + g * g)).sum(axis=0)
    return y


def lorentz_sum(x, sticks, fwhm_ppm):
    """ローレンツ型の足し合わせ。線を細かい格子 (間隔 <= 半値半幅 / 6) に置いて、FFT で畳み込む。
    線の数 × 点の数が 5e7 未満なら厳密な和を使う。それ以上なら、厳密な和との差は最大値の 0.5% 未満。x の範囲の外 (半値半幅の 200 倍まで) の線も含める。"""
    x = np.asarray(x, float)
    fx, fw = np.asarray(sticks[0], float), np.asarray(sticks[1], float)
    if len(fx) * len(x) < 5e7:                          # 線が少ないときは、厳密な和の方が速く、誤差もない
        return lorentz_sum_exact(x, (fx, fw), fwhm_ppm)
    g = fwhm_ppm / 2
    dx0 = (x[-1] - x[0]) / (len(x) - 1)
    step = dx0 / int(np.ceil(dx0 / (g / 6))) if dx0 > g / 6 else dx0
    pad = 200 * g
    x0 = x[0] - pad
    nf = int(np.ceil((x[-1] + pad - x0) / step)) + 2
    keep = (fx >= x0) & (fx <= x0 + (nf - 2) * step)
    pos = (fx[keep] - x0) / step
    i = np.floor(pos).astype(int); fr = pos - i
    h = np.bincount(i, weights=fw[keep] * (1 - fr), minlength=nf + 1)[:nf] + np.bincount(i + 1, weights=fw[keep] * fr, minlength=nf + 1)[:nf]
    t = np.arange(-(nf - 1), nf) * step
    k = g * g / (t * t + g * g)
    n = 1 << int(np.ceil(np.log2(3 * nf)))
    conv = np.fft.irfft(np.fft.rfft(h, n) * np.fft.rfft(k, n), n)[nf - 1:2 * nf - 1]
    return np.interp(x, x0 + step * np.arange(nf), conv)


def plot(mol0, peaks, path, title, mhz, sticks=None, fwhm_hz=1.2, mults=None):
    """上段: 全体, 下段: ピークの塊ごとの拡大図 (シフトと H 数つき)。"""
    import matplotlib
    matplotlib.use('Agg')
    matplotlib.rcParams['font.family'] = ['Yu Gothic', 'Meiryo', 'MS Gothic', 'DejaVu Sans']
    import matplotlib.pyplot as plt
    if sticks is None:
        sticks = (np.array([d for d, _, _, _ in peaks]), np.array([float(n) for _, n, _, _ in peaks]))
    fw_ppm = fwhm_hz / mhz
    mlab = {round(p[0], 6): m for p, m in zip(peaks, mults)} if mults else {}
    hi = max(max(p[0] for p in peaks) + 1.0, 10.0)
    lo = min(-0.5, min(p[0] for p in peaks) - 0.5)
    x = np.linspace(lo, hi, 12000)
    y = lorentz_sum(x, sticks, fw_ppm)

    # ピークの塊: 線の位置が 0.5 ppm 以内ならひとまとめ
    pos = np.sort(np.array([p[0] for p in peaks]))
    clusters, cur = [], [pos[0]]
    for v in pos[1:]:
        if v - cur[-1] > 0.5:
            clusters.append(cur); cur = []
        cur.append(v)
    clusters.append(cur)
    wins = []
    for c in clusters:
        in_c = [p for p in peaks if min(c) - 1e-9 <= p[0] <= max(c) + 1e-9]
        fx = sticks[0][(sticks[0] > min(c) - 0.3) & (sticks[0] < max(c) + 0.3)]
        wins.append((min(fx.min(), min(c)) - 0.04, max(fx.max(), max(c)) + 0.04, in_c))
    wins.sort(key=lambda w: -w[0])                                  # 低磁場 (大きい ppm) を左に

    nz = len(wins)
    fig = plt.figure(figsize=(max(10, 3.2 * nz), 7))
    nmax = max(len(w[2]) for w in wins)                              # 1 つの塊の中のピーク数 (ラベルの行数)
    fig.set_size_inches(max(10, 3.2 * nz), 8.2 + 0.5 * max(0, nmax - 4))
    gs = fig.add_gridspec(3, nz, height_ratios=[0.55, 1, 1.15 + 0.12 * max(0, nmax - 4)], hspace=0.45)   # 1 行目: 構造式, 2 行目: 全体, 3 行目: 拡大図
    ax = fig.add_subplot(gs[1, :])
    ax.plot(x, y, color='black', lw=0.9)
    ax.set_xlim(hi, lo); ax.set_ylim(0, y.max() * 1.35)
    ax.set_yticks([]); ax.set_xlabel('δ (ppm)')
    for sp in ('top', 'right', 'left'):
        ax.spines[sp].set_visible(False)
    ax.set_title(f'{title}   (predicted 1H, {mhz:.0f} MHz, empirical J)', fontsize=10)
    for k, (w0, w1, _) in enumerate(wins):
        ax.axvspan(w0, w1, color='tab:blue', alpha=0.12, lw=0)
    hd = fig.add_subplot(gs[0, :])                                   # 構造式は専用の行に置く (スペクトルに重ならない)
    hd.imshow(Draw.MolToImage(Chem.RemoveHs(mol0), size=(640, 170))); hd.axis('off')

    for k, (w0, w1, in_c) in enumerate(wins):
        a = fig.add_subplot(gs[2, k])
        xz = np.linspace(w0, w1, 4000)
        yz = lorentz_sum(xz, sticks, fw_ppm)
        a.plot(xz, yz, color='black', lw=0.9)
        step = min(0.075, 0.74 / max(1, len(in_c)))                  # ラベルの行間 (軸の高さに対する割合)
        top = min(0.05 + step * len(in_c) + 0.04, 0.8)               # ラベルが占める上側の割合
        a.set_xlim(w1, w0); a.set_ylim(0, yz.max() / (1 - top))
        a.set_yticks([])
        for sp in ('top', 'right', 'left'):
            a.spines[sp].set_visible(False)
        a.xaxis.set_major_locator(plt.MaxNLocator(4))
        a.tick_params(labelsize=8)
        for i, (d, n, _, ex) in enumerate(sorted(in_c, key=lambda p: -p[0])):
            m = mlab.get(round(d, 6))
            a.annotate(f'{d:.2f}' + ('*' if ex else '') + f' ({n}H' + (f', {mult_text(*m, short=True)}' if m else '') + ')', (d, 0),
                       xytext=(0.5, 0.98 - step * i), textcoords='axes fraction', ha='center', va='top', fontsize=8,
                       bbox=dict(fc='white', ec='none', pad=0.6, alpha=0.85), zorder=5)
            a.axvline(d, ymax=max(0.05, 0.98 - step * i - 0.06), color='gray', lw=0.5, ls=':', zorder=1)
    if any(p[3] for p in peaks):
        fig.text(0.01, 0.01, '* O-H/N-H 等: 溶媒・濃度で大きく変わるため参考値', fontsize=8)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def csv_files():
    """化合物の一覧: nmr_local.csv (自分の化合物。.gitignore で除外) があれば先に、続いて nmr.csv (サンプル)。"""
    return [p for p in (os.path.join(HERE, 'nmr_local.csv'), os.path.join(HERE, 'nmr.csv')) if os.path.exists(p)]


def resolve(target):
    """nmr_local.csv / nmr.csv / nmr_local.csv の Compound_ID ならその SMILES、そうでなければ SMILES とみなす。"""
    import csv
    for p in csv_files():
        for r in csv.DictReader(open(p, encoding='utf-8')):
            if r['Compound_ID'] == target:
                return target, r['SMILES'], r.get('Solvent') or None
    return None, target, None


def known_ids():
    import csv
    seen = []
    for p in csv_files():
        for r in csv.DictReader(open(p, encoding='utf-8')):
            if r['Compound_ID'] not in seen:
                seen.append(r['Compound_ID'])
    return seen


def run_one(cfg, args, target):
    cid, smiles, csv_solv = resolve(target)
    solv = (args.solvent or csv_solv or 'CDCl3')
    if solv.lower() not in SOLVENT:
        sys.exit(f'エラー: 溶媒 {solv} は未対応です。使えるもの: ' + ', '.join(SOLVENT))
    alpb, cpcm = SOLVENT[solv.lower()]
    mol0 = Chem.MolFromSmiles(smiles)
    if mol0 is None:
        sys.exit(f'エラー: SMILES として読めません: {smiles}')
    name = args.name or cid or 'mol_' + hashlib.md5(smiles.encode()).hexdigest()[:8]
    mol = Chem.AddHs(mol0)
    symbols = [at.GetSymbol() for at in mol.GetAtoms()]
    calc = args.nuc_list
    two_d = [n for n in calc if n in TWOD]
    # 必要な核種だけを計算する (2D: すべて 1H が必要。HSQC / HMBC は 13C も)
    need = set(n for n in calc if n in NUC)
    if two_d:
        need.add('1H')
    if any(k in two_d for k in ('HSQC', 'HMQC', 'HMBC')):
        need.add('13C')
    missing = [n for n in NUC if n in need and NUC[n]['el'] not in symbols]
    for n in missing:
        print(f'  {n}: この分子には {NUC[n]["el"]} 原子がないので、スキップします')
    need -= set(missing)
    two_d = [n for n in two_d if '1H' in need and not (n in ('HSQC', 'HMQC', 'HMBC') and '13C' not in need)]
    nuc1d = [n for n in NUC if n in calc and n in need]
    if not need or not (nuc1d or two_d):
        sys.exit('エラー: 指定した計算に必要な原子が分子にありません。')
    want_j = ('1H' in nuc1d and not args.no_j) or 'COSY' in two_d or 'TOCSY' in two_d
    auto = str(cfg['method']).lower() == 'auto'
    cfg = resolve_method(cfg, need)
    why = ('; 13C を含むので PBE0 を自動選択' if '13C' in need else '; 13C を含まないので PBE を自動選択') if auto else ''
    print(f'{name}: {smiles}  ({solv}, {cfg["method"]}/{cfg["basis"]}{why})')
    print(f'  コア数 {cfg["nprocs"]} (自動検出), ORCA メモリ {cfg["maxcore_mb"]} MB/コア')
    work = os.path.join(HERE, 'work', f"{name}_{solv.lower()}")
    confs_xyz = []
    res = compute_all(cfg, mol, work, alpb, cpcm, args.nconf, max_orca=args.orca_conf or cfg['orca_conf'],
                      want_j=want_j, conf_out=confs_xyz if ('NOESY' in two_d or 'ROESY' in two_d) else None)
    sig_all, jmat = res if want_j else (res, None)
    os.makedirs(os.path.join(HERE, 'out'), exist_ok=True)
    last_png = None
    peaks_by = {}
    for nuc in [n for n in NUC if n in need]:
        el = NUC[nuc]['el']
        sigma = {i: v for i, v in sig_all.items() if symbols[i] == el}
        ref = reference_sigma(cfg, nuc, alpb, cpcm)
        peaks = peaks_from(mol, sigma, ref, el, cfg['scale'].get(nuc))
        peaks_by[nuc] = peaks
        if nuc not in nuc1d:
            continue
        base = os.path.join(HERE, 'out', f'{name}_{nuc}')
        mults = None
        if nuc == '1H' and jmat is not None and not args.no_j:
            mults = multiplets(mol, peaks, jmat, args.mhz)
        with open(base + '.csv', 'w', encoding='utf-8') as f:
            f.write('shift_ppm,n_atoms,atom_indices,exchangeable' + (',multiplicity,J_Hz' if mults else '') + '\n')
            for k, (d, n, idxs, ex) in enumerate(peaks):
                f.write(f'{d:.3f},{n},{" ".join(map(str, idxs))},{int(ex)}')
                if mults:
                    f.write(f',{mults[k][0]},{" ".join(f"{j:.1f}" for j in mults[k][1])}')
                f.write('\n')
        gtab = []
        if nuc == '1H':
            sticks = None
            if jmat is not None and not args.no_j:
                fx, fw, gtab = build_spectrum(mol, peaks, jmat, args.mhz)
                sticks = (fx, fw)
            if gtab:
                with open(os.path.join(HERE, 'out', f'{name}_J.csv'), 'w', encoding='utf-8') as f:
                    f.write('shift_A_ppm,shift_B_ppm,J_Hz,n_pairs\n')
                    for a_, b_, v, n in gtab:
                        f.write(f'{a_:.3f},{b_:.3f},{v:.2f},{n}\n')
            plot(mol0, peaks, base + '.png', name, args.mhz, sticks, mults=mults)
        else:
            plot_nuc(mol0, peaks, base + '.png', name, nuc, args.mhz)
        print(f'\n  {nuc}  δ (ppm, 基準 {NUC[nuc]["refname"]})   原子数' + ('   多重度, J' if mults else ''))
        for k, (d, n, _, ex) in enumerate(peaks):
            print(f'  {d:9.2f}   {n}' + (f'   {mult_text(*mults[k])}' if mults and not ex else '') + ('   (O-H/N-H: 参考値)' if ex else ''))
        if mults:
            print('\n  ' + report_1H(peaks, mults, args.mhz))
        if gtab:
            print('\n  J 結合 (Hz, 経験式・配座/回転平均):')
            for a_, b_, v, n in gtab:
                print(f'    {a_:5.2f} - {b_:5.2f} ppm   {v:6.2f}')
        print(f'  画像: {base}.png\n  表:   {base}.csv')
        last_png = base + '.png'

    # 2 次元
    if two_d:
        pH = peaks_by['1H']
        pC = peaks_by.get('13C')
        tH = [(d, float(n)) for d, n, _, ex in pH if not ex]
        tC = [(d, float(n)) for d, n, _, _ in pC] if pC else None
        if any(k in two_d for k in ('COSY', 'TOCSY')):
            _, group, _, Jd = pair_couplings(mol, pH, jmat)
        for kind in two_d:
            if kind == 'COSY':
                pts, nx, ny, tx, ty = cosy_points(pH, Jd, group, args.cosy_jmin), '1H', '1H', tH, tH
            elif kind == 'TOCSY':
                pts, nx, ny, tx, ty = tocsy_points(pH, Jd, group, args.cosy_jmin), '1H', '1H', tH, tH
            elif kind == 'NOESY':
                pts, nx, ny, tx, ty = noesy_points(pH, confs_xyz), '1H', '1H', tH, tH
            elif kind == 'ROESY':
                pts, nx, ny, tx, ty = noesy_points(pH, confs_xyz, cls='roe'), '1H', '1H', tH, tH
            elif kind == 'HMQC':
                pts, nx, ny, tx, ty = hsqc_points(mol, pH, pC, edited=False), '1H', '13C', tH, tC
            elif kind == 'HSQC':
                pts, nx, ny, tx, ty = hsqc_points(mol, pH, pC), '1H', '13C', tH, tC
            else:
                pts, nx, ny, tx, ty = hmbc_points(mol, pH, pC), '1H', '13C', tH, tC
            base = os.path.join(HERE, 'out', f'{name}_{kind}')
            with open(base + '.csv', 'w', encoding='utf-8') as f:
                f.write(f'x_{nx}_ppm,y_{ny}_ppm,intensity,type,note\n')
                for x, y, w, cls, note in pts:
                    f.write(f'{x:.3f},{y:.3f},{w:.3f},{cls},{note}\n')
            plot_2d(kind, pts, nx, ny, base + '.png', name, mol0, tx, ty)
            ncross = sum(1 for p in pts if p[3] != 'diag')
            print(f'\n  {kind}: 相関ピーク {ncross} 個')
            print(f'  画像: {base}.png\n  表:   {base}.csv')
            last_png = base + '.png'
    if last_png and not args.no_open and os.name == 'nt':
        try:
            os.startfile(last_png)
        except OSError:
            pass


def main():
    try:
        sys.stdout.reconfigure(errors='replace')            # コンソールの文字コードで表示できない文字があっても落とさない
    except Exception:
        pass
    ap = argparse.ArgumentParser(description='xtb + ORCA による NMR 予測 (1H, 13C, 31P, 19F, 11B)。引数なしで起動すると対話モード')
    ap.add_argument('target', nargs='?', help='SMILES または nmr.csv / nmr_local.csv の Compound_ID (省略すると入力を求める)')
    ap.add_argument('--name', help='出力名 (省略時は ID か SMILES のハッシュ)')
    ap.add_argument('--solvent', help='CDCl3 / DMSO-d6 / CD3OD / D2O / acetone-d6 / C6D6 (既定 CDCl3)')
    ap.add_argument('--nconf', type=int, default=5, help='xtb で最適化する配座数 (既定 5)')
    ap.add_argument('--nprocs', type=int, help='使うコア数 (省略時は論理コア数を自動検出して全部使う)')
    ap.add_argument('--orca-conf', type=int, help='ORCA (DFT) で計算する配座数 (既定 1。増やすと遅いが柔軟な分子で精度が上がる)')
    ap.add_argument('--calc', '--nuc', dest='nuc', default='1H', help='計算するもの (カンマ区切り): 核種 1H (既定), 13C, 31P, 19F, 11B / 2 次元 COSY, HSQC, HMQC, HMBC, TOCSY, NOESY, ROESY / all = 全核種, 2D = 全 2 次元 (例: --calc 1H,13C,COSY,HSQC)')
    ap.add_argument('--cosy-jmin', type=float, default=2.0, help='COSY / TOCSY で相関を出す J の下限 Hz (既定 2.0)')
    ap.add_argument('--no-j', action='store_true', help='J 結合を計算しない (速い。線は分裂しない)')
    ap.add_argument('--no-open', action='store_true', help='完成した画像を自動で開かない')
    ap.add_argument('--mhz', type=float, default=400.0, help='表示する装置周波数 (既定 400)')
    args = ap.parse_args()
    try:
        args.nuc_list = parse_calc(args.nuc)
    except ValueError as e:
        sys.exit(f'エラー: 未対応の指定: {e}  (使えるもの: ' + ', '.join(list(NUC) + TWOD) + ')')

    cfg = load_config()
    if args.nprocs:
        cfg['nprocs'] = args.nprocs
    cfg = resolve_resources(cfg)

    if args.target:
        run_one(cfg, args, args.target)
        return
    # 対話モード: 計算するもの -> SMILES か Compound_ID の順に聞く。計算するものが空なら 1H。SMILES が空なら終了
    ids = known_ids()
    print('NMR 予測 (xtb + ORCA)')
    if ids:
        print('nmr.csv / nmr_local.csv の Compound_ID: ' + ', '.join(ids))
    print(f'溶媒: {args.solvent or "CDCl3"}   (変えるには --solvent を付けて起動)\n')
    while True:
        try:
            n = input('計算するもの (1H, 13C, 31P, 19F, 11B / COSY, HSQC, HMQC, HMBC, TOCSY, NOESY, ROESY / カンマ区切り / all=全核種, 2D=全2次元。空 Enter で 1H): ').strip()
            try:
                nl = parse_calc(n)
            except ValueError as e:
                print(f'未対応の指定: {e}'); continue
            t = input('SMILES または Compound_ID (空 Enter で終了): ').strip().strip('"').strip("'")
        except (EOFError, KeyboardInterrupt):
            print(); break
        if not t:
            break
        args.name = None
        args.nuc_list = nl
        try:
            run_one(cfg, args, t)
        except SystemExit as e:                     # エラーで落とさず、次の入力に戻る
            print(e.code if isinstance(e.code, str) else '')
        print()


if __name__ == '__main__':
    main()
