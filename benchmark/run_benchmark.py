"""手法ごとにベンチマーク分子を計算し、結果の CSV を benchmark/results/<手法>/ に保存する。
使い方: python run_benchmark.py <PBE|PBE0|r2SCAN0> [分子ID ...]   (xtb の構造は work/ で手法間に共有される)"""
import os, sys, shutil, time
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
import nmr
from refdata import MOLS, HETERO, MOLS_DMSO, HETERO2

METHODS = {
    'PBE':     {'method': 'PBE',     'basis': 'def2-SVP', 'orca_extra': 'def2/J'},
    'PBE0':    {'method': 'PBE0',    'basis': 'def2-SVP', 'orca_extra': ''},
    'PBE0_pcSseg1': {'method': 'PBE0', 'basis': 'pcSseg-1', 'orca_extra': ''},     # NMR 用の基底関数 (def2-SVP と同程度の大きさ)
    'r2SCAN0': {'method': 'r2SCAN0', 'basis': 'def2-SVP', 'orca_extra': 'RIJCOSX def2/J'},
}
name = sys.argv[1]
only = sys.argv[2:]
SET = 'main'                                   # main / dmso / hetero2  (例: run_benchmark.py PBE0 --set=dmso)
if only and only[0].startswith('--set='):
    SET = only.pop(0).split('=')[1]
SOLV = 'DMSO-d6' if SET == 'dmso' else None
_orig = nmr.load_config
def _cfg():
    c = _orig(); c.update(METHODS[name]); return c
nmr.load_config = _cfg

if SET == 'dmso':
    jobs = [(i, s, ['1H', '13C']) for i, (s, _) in MOLS_DMSO.items()]; HET = {}
elif SET == 'hetero2':
    jobs = []; HET = HETERO2
else:
    jobs = [(i, s, ['1H', '13C']) for i, (s, _) in MOLS.items()]; HET = HETERO
for nuc, d in HET.items():
    for i, (s, _) in d.items():
        jobs.append((i, s, [nuc]))
dest = os.path.join(HERE, 'results', name + ({'main': '', 'dmso': '_DMSO', 'hetero2': ''}[SET]))
os.makedirs(dest, exist_ok=True)
for i, s, nucs in jobs:
    if only and i not in only:
        continue
    if all(os.path.exists(os.path.join(dest, f'{i}_{n}.csv')) for n in nucs):
        continue
    t = time.time()
    sys.argv = ['nmr.py', s, '--name', ('bmD_' if SET == 'dmso' else 'bm_') + i, '--calc', ','.join(nucs), '--no-j', '--no-open', '--nconf', '5'] + (['--solvent', SOLV] if SOLV else [])
    try:
        nmr.main()
    except SystemExit as e:
        print('FAILED', i, e.code); continue
    for n in nucs:
        shutil.copy(os.path.join(ROOT, 'out', f'{"bmD_" if SET == "dmso" else "bm_"}{i}_{n}.csv'), os.path.join(dest, f'{i}_{n}.csv'))
    print(f'DONE {name} {i} {time.time() - t:.0f}s', flush=True)
