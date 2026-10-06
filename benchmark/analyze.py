"""results/<手法>/ の予測と文献値を比べ、results/summary.md を書く。
比較: 予測と文献の信号を原子ごとに展開し、ppm の順に並べて 1 対 1 で対応づける (帰属なし)。交換性 H は除く。"""
import os, csv, math, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from refdata import MOLS, HETERO
METHODS = ['PBE', 'PBE0', 'r2SCAN0']

def pred(method, i, nuc):
    p = os.path.join(HERE, 'results', method, f'{i}_{nuc}.csv')
    if not os.path.exists(p):
        return None
    out = []
    for r in csv.DictReader(open(p, encoding='utf-8')):
        if int(r.get('exchangeable', 0) or 0):
            continue
        n = int(r.get('nH') or r.get('n_atoms'))
        out += [float(r['shift_ppm'])] * n
    return sorted(out)

def expand(sig):
    return sorted(s for s, n in sig for _ in range(n))

def stats(err):
    e = np.array(err)
    return dict(n=len(e), mae=np.abs(e).mean(), rmse=math.sqrt((e**2).mean()), bias=e.mean(), mx=np.abs(e).max())

def collect(method, nuc):
    err, per = [], {}
    for i, (_, d) in MOLS.items():
        p = pred(method, i, nuc)
        if p is None:
            continue
        x = expand(d[nuc])
        if len(p) != len(x):
            per[i] = None; continue
        e = [a - b for a, b in zip(p, x)]
        err += e; per[i] = e
    return err, per

L = []
L.append('# ベンチマーク結果 (CDCl3 の CPCM, 補正なし, 基準 TMS)\n')
L.append('比較は、予測と文献値を原子ごとに展開して ppm の順に並べ、1 対 1 で対応づけた (帰属なし)。誤差 = 予測 − 文献。\n')
for nuc in ['1H', '13C']:
    L.append(f'\n## {nuc}\n')
    L.append('| 手法 | 原子数 | MAE (ppm) | RMSE | 平均 (偏り) | 最大 |')
    L.append('|---|---|---|---|---|---|')
    allres = {}
    for m in METHODS:
        err, per = collect(m, nuc)
        allres[m] = per
        if err:
            s = stats(err)
            L.append(f"| {m} | {s['n']} | {s['mae']:.3f} | {s['rmse']:.3f} | {s['bias']:+.3f} | {s['mx']:.2f} |")
    L.append(f'\n分子ごとの MAE (ppm, {nuc}):\n')
    L.append('| 分子 | ' + ' | '.join(METHODS) + ' |')
    L.append('|---|' + '---|' * len(METHODS))
    for i in MOLS:
        row = []
        for m in METHODS:
            e = allres[m].get(i)
            row.append('-' if e is None else f'{np.abs(e).mean():.2f}')
        L.append(f'| {i} | ' + ' | '.join(row) + ' |')
    # 構造による層別 (13C: sp3 / sp2・sp / C=O)
    if nuc == '13C':
        L.append('\n13C を炭素の種類で分けた MAE (ppm):\n')
        L.append('| 手法 | sp3 (文献 < 80) | sp2・sp (80-180) | C=O (> 180) |')
        L.append('|---|---|---|---|')
        for m in METHODS:
            bins = {'a': [], 'b': [], 'c': []}
            for i, (_, d) in MOLS.items():
                e = allres[m].get(i)
                if e is None: continue
                x = expand(d['13C'])
                for xv, ev in zip(x, e):
                    bins['a' if xv < 80 else 'b' if xv < 180 else 'c'].append(abs(ev))
            f = lambda v: f'{np.mean(v):.2f} (n={len(v)})' if v else '-'
            L.append(f"| {m} | {f(bins['a'])} | {f(bins['b'])} | {f(bins['c'])} |")
# 他の核種
L.append('\n## 19F / 31P / 11B (少数の分子。文献値は目安)\n')
L.append('| 核種 | 分子 | 文献 | ' + ' | '.join(METHODS) + ' |')
L.append('|---|---|---|' + '---|' * len(METHODS))
for nuc, d in HETERO.items():
    for i, (_, ref) in d.items():
        row = []
        for m in METHODS:
            p = pred(m, i, nuc)
            row.append('-' if not p else f'{np.mean(p):.1f} ({np.mean(p) - ref:+.1f})')
        L.append(f'| {nuc} | {i} | {ref} | ' + ' | '.join(row) + ' |')
open(os.path.join(HERE, 'results', 'summary.md'), 'w', encoding='utf-8').write('\n'.join(L) + '\n')
print('\n'.join(L))
