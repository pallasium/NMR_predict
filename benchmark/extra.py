"""追加セット (19F / 31P の小分子、DMSO-d6) の集計。results/extra.md に書く。"""
import os, sys, csv
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np
from refdata import HETERO2, HETERO2_VERIFIED, MOLS_DMSO, MOLS
R = os.path.join(HERE, 'results')
def one(path):
    if not os.path.exists(path): return None
    v = [float(r['shift_ppm']) for r in csv.DictReader(open(path, encoding='utf-8')) if not int(r.get('exchangeable', 0) or 0)]
    return float(np.mean(v)) if v else None
def sig(path, per):
    out = []
    for r in csv.DictReader(open(path, encoding='utf-8')):
        if int(r.get('exchangeable', 0) or 0): continue
        out += [float(r['shift_ppm'])] * int(r.get('nH') or r.get('n_atoms'))
    return sorted(out)
M = ['PBE', 'PBE0']
L = ['# 追加セットの結果 (PBE と PBE0。r2SCAN0 は時間節約のため追加分では計算していない)\n', '## 19F / 31P (照合済みの標準物質の値。溶媒の記載なし)\n',
     '| 核種 | 分子 | 文献 | PBE (差) | PBE0 (差) |', '|---|---|---|---|---|']
bias = {(n, m): [] for n in ('19F', '31P') for m in M}
for nuc in ('19F', '31P'):
    for i, (_, ref) in HETERO2[nuc].items():
        row = []
        for m in M:
            v = one(os.path.join(R, m, f'{i}_{nuc}.csv'))
            row.append('-' if v is None else f'{v:.1f} ({v - ref:+.1f})')
            if v is not None: bias[(nuc, m)].append(v - ref)
        L.append(f'| {nuc} | {i} | {ref} | ' + ' | '.join(row) + ' |')
L.append('\n| 核種 | 手法 | 分子数 | MAE | 平均の偏り |'); L.append('|---|---|---|---|---|')
for (nuc, m), e in bias.items():
    if e: L.append(f'| {nuc} | {m} | {len(e)} | {np.abs(e).mean():.1f} | {np.mean(e):+.1f} |')
L.append('\n## DMSO-d6 (CPCM(DMSO), Fulmer ら 2010 の値。3 分子)\n'); L.append('| 核種 | 手法 | DMSO-d6 の MAE | (同じ 3 分子の CDCl3 の MAE) |'); L.append('|---|---|---|---|')
names = ['ethyl_acetate', 'DMF', 'toluene']
for nuc in ('1H', '13C'):
    for m in M:
        for tag, d, mols in (('D', MOLS_DMSO, None),):
            pass
        e_d, e_c = [], []
        for i in names:
            for sol, E, ref in (('D', e_d, MOLS_DMSO), ('C', e_c, MOLS)):
                p = os.path.join(R, m + ('_DMSO' if sol == 'D' else ''), f'{i}_{nuc}.csv')
                x = sorted(s for s, n in ref[i][1][nuc] for _ in range(n)); pr = sig(p, None)
                E += [a - b for a, b in zip(pr, x)] if len(pr) == len(x) else []
        L.append(f'| {nuc} | {m} | {np.abs(e_d).mean():.3f} (n={len(e_d)}) | {np.abs(e_c).mean():.3f} |')
open(os.path.join(R, 'extra.md'), 'w', encoding='utf-8').write('\n'.join(L) + '\n')
