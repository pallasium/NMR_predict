"""PBE0 の基底関数の比較: def2-SVP と pcSseg-1。results/basis.md に書く。"""
import os, sys, csv
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np
from refdata import MOLS, VERIFIED, HETERO2
R = os.path.join(HERE, 'results'); D = ['PBE0', 'PBE0_pcSseg1']
def sig(p):
    out = []
    for r in csv.DictReader(open(p, encoding='utf-8')):
        if int(r.get('exchangeable', 0) or 0): continue
        out += [float(r['shift_ppm'])] * int(r.get('nH') or r.get('n_atoms'))
    return sorted(out)
L = ['# PBE0 の基底関数: def2-SVP と pcSseg-1 (CDCl3, 補正なし)\n']
for label, names in (('有機 8 分子', list(MOLS)), ('Fulmer の表で照合した 5 分子', VERIFIED)):
    L += [f'\n## {label}\n', '| 核種 | 基底関数 | MAE (ppm) | 平均の偏り | 原子数 |', '|---|---|---|---|---|']
    for nuc in ('1H', '13C'):
        for d in D:
            e = []
            for i in names:
                x = sorted(s for s, n in MOLS[i][1][nuc] for _ in range(n)); p = sig(os.path.join(R, d, f'{i}_{nuc}.csv'))
                e += [a - b for a, b in zip(p, x)]
            L.append(f'| {nuc} | {d.replace("PBE0_pcSseg1", "pcSseg-1").replace("PBE0", "def2-SVP")} | {np.abs(e).mean():.3f} | {np.mean(e):+.3f} | {len(e)} |')
L += ['\n## 19F / 31P / 11B (差 = 予測 − 文献, ppm)\n', '| 核種 | 分子 | 文献 | def2-SVP | pcSseg-1 |', '|---|---|---|---|---|']
rows = [('19F', 'fluorobenzene', -113.15), ('19F', 'benzotrifluoride', -63.72), ('31P', 'Me3PO', 36.2), ('31P', 'PMe3', -62.0), ('11B', 'B(OMe)3', 18.1)]
eh = {d: [] for d in D}
for nuc, i, ref in rows:
    cells = []
    for d in D:
        p = os.path.join(R, d, f'{i}_{nuc}.csv'); v = np.mean(sig(p)); cells.append(f'{v:.1f} ({v - ref:+.1f})')
        if nuc != '11B': eh[d].append(v - ref)
    L.append(f'| {nuc} | {i} | {ref} | ' + ' | '.join(cells) + ' |')
L.append('\n19F / 31P の 4 分子の MAE: def2-SVP ' + f'{np.abs(eh["PBE0"]).mean():.1f}' + ' ppm / pcSseg-1 ' + f'{np.abs(eh["PBE0_pcSseg1"]).mean():.1f}' + ' ppm')
open(os.path.join(R, 'basis.md'), 'w', encoding='utf-8').write('\n'.join(L) + '\n')
