"""分子を単位にしたブートストラップで、手法間の差が偶然の範囲かを見る。results/stats.md に書く。
1H / 13C の誤差は analyze.py と同じ (予測と文献を ppm の順に並べて 1 対 1)。分子内の原子は相関するので、原子ではなく分子を再標本化する。"""
import os, sys, csv
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np
from refdata import MOLS, VERIFIED
METHODS = ['PBE', 'PBE0', 'r2SCAN0']

def errs(method, nuc, names):
    out = {}
    for i in names:
        p = os.path.join(HERE, 'results', method, f'{i}_{nuc}.csv')
        pr = []
        for r in csv.DictReader(open(p, encoding='utf-8')):
            if int(r.get('exchangeable', 0) or 0): continue
            pr += [float(r['shift_ppm'])] * int(r.get('nH') or r.get('n_atoms'))
        x = sorted(s for s, n in MOLS[i][1][nuc] for _ in range(n))
        out[i] = np.array(sorted(pr)) - np.array(x)
    return out

rng = np.random.default_rng(0)
L = ['# 手法間の差の統計 (分子を単位にしたブートストラップ, 10000 回)\n']
for label, names in [('全 8 分子', list(MOLS)), ('Fulmer の表で照合した 5 分子', [m for m in VERIFIED if m in MOLS])]:
    L.append(f'\n## {label}\n')
    for nuc in ['1H', '13C']:
        E = {m: errs(m, nuc, names) for m in METHODS}
        n = len(names)
        def mae(m, idx): return np.concatenate([np.abs(E[m][names[k]]) for k in idx]).mean()
        L.append(f'\n### {nuc}\n'); L.append('| 手法 | MAE (ppm) | 95% 区間 |'); L.append('|---|---|---|')
        boots = {m: [] for m in METHODS}
        idxs = [rng.integers(0, n, n) for _ in range(10000)]
        for m in METHODS:
            boots[m] = np.array([mae(m, ix) for ix in idxs])
            L.append(f'| {m} | {mae(m, range(n)):.3f} | {np.percentile(boots[m], 2.5):.3f} – {np.percentile(boots[m], 97.5):.3f} |')
        L.append('\n| 比較 (A − B の MAE) | 差 (ppm) | 95% 区間 | A がよい確率 |'); L.append('|---|---|---|---|')
        for a, b in [('PBE0', 'PBE'), ('r2SCAN0', 'PBE'), ('PBE0', 'r2SCAN0')]:
            d = boots[a] - boots[b]
            L.append(f'| {a} − {b} | {mae(a, range(n)) - mae(b, range(n)):+.3f} | {np.percentile(d, 2.5):+.3f} – {np.percentile(d, 97.5):+.3f} | {(d < 0).mean():.0%} |')
open(os.path.join(HERE, 'results', 'stats.md'), 'w', encoding='utf-8').write('\n'.join(L) + '\n')
