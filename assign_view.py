"""どの原子がどのピークか、を見えるようにする (NMR_fast / NMR_superfast 共通。2 つのフォルダに同じものを置く)。

  <out>/<name>_<核種>_assign.png   構造式の原子に番号と色、スペクトルのピークに同じ番号と色 (案 1)
  <out>/<name>_<核種>_assign.html  ブラウザで開く。原子をクリックするとピークが光り、ピークをクリックすると原子が光る (案 2)

番号は RDKit の原子 index (0 始まり)。`out/*.csv` の atom_indices 列と同じ。
1H は、H が結合している重原子の番号で表す (CH3 の H 3 つは、その C の番号 1 つ)。
peaks は nmr.py の [(shift ppm, 原子数, [原子 index], 交換性か)]。"""
import html, json, os
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D

COLORS = ['#e6194b', '#3cb44b', '#4363d8', '#f58231', '#911eb4', '#0aa5b5', '#c2185b', '#808000',
          '#795548', '#008080', '#9a6324', '#6a3d9a', '#b15928', '#2e7d32', '#1565c0', '#ef6c00']
FWHM = {'1H': 0.03, '13C': 0.6, '31P': 1.0, '19F': 1.0, '11B': 1.0}      # 表示用の線幅 (ppm)。見やすさのため、実際より太い


def _hosts(mol, idxs):
    """ピークの原子 index -> 構造式に描かれている原子 (H なら結合先の重原子)。"""
    out = []
    for i in idxs:
        a = mol.GetAtomWithIdx(i)
        h = a.GetNeighbors()[0].GetIdx() if a.GetSymbol() == 'H' and a.GetDegree() == 1 else i
        if h not in out:
            out.append(h)
    return out


def _label(nuc, hosts, ex):
    return ('' if nuc != '1H' else 'H on ') + ','.join(map(str, hosts)) + (' *' if ex else '')


def _prepare(mol0, peaks, nuc):
    mol = Chem.AddHs(mol0)
    molh = Chem.RemoveHs(mol0)
    rdDepictor.Compute2DCoords(molh)
    info = []
    for k, (d, n, idxs, ex) in enumerate(peaks):
        hosts = _hosts(mol, idxs)
        info.append(dict(k=k, shift=float(d), n=int(n), ex=bool(ex), atoms=hosts, label=_label(nuc, hosts, ex),
                         color=COLORS[k % len(COLORS)]))
    return molh, info


def _trace(shifts, weights, lo, hi, fwhm, npts=4000):
    x = np.linspace(lo, hi, npts)
    g = fwhm / 2
    y = sum(w * g * g / ((x - s) ** 2 + g * g) for s, w in zip(shifts, weights))
    return x, y


def write_png(mol0, peaks, nuc, path, title, mhz=None):
    import matplotlib
    matplotlib.use('Agg')
    matplotlib.rcParams['font.family'] = ['Yu Gothic', 'Meiryo', 'MS Gothic', 'DejaVu Sans']
    import matplotlib.pyplot as plt
    from io import BytesIO
    from PIL import Image
    molh, info = _prepare(mol0, peaks, nuc)
    hl, hc = [], {}
    for p in info:
        for a in p['atoms']:
            if a not in hc:
                hl.append(a)
                hc[a] = tuple(int(p['color'][i:i + 2], 16) / 255 for i in (1, 3, 5))
    try:
        dr = rdMolDraw2D.MolDraw2DCairo(900, 420)
    except AttributeError:
        raise RuntimeError('RDKit の Cairo 描画が使えません')
    dr.drawOptions().highlightRadius = 0.3
    rdMolDraw2D.PrepareAndDrawMolecule(dr, molh, highlightAtoms=hl,
                                       highlightAtomColors={a: hc[a] + (0.3,) for a in hl},
                                       highlightBonds=[])
    dr.FinishDrawing()
    img = Image.open(BytesIO(dr.GetDrawingText()))
    pos = {a.GetIdx(): dr.GetDrawCoords(a.GetIdx()) for a in molh.GetAtoms()}
    col = {a: p['color'] for p in info for a in p['atoms']}

    shifts = [p['shift'] for p in info]
    span = max(shifts) - min(shifts)
    pad = max(0.08 * span, 0.5 if nuc == '1H' else 8)
    lo, hi = min(shifts) - pad, max(shifts) + pad
    if nuc == '1H':
        lo, hi = min(lo, -0.3), max(hi, 10.3)
    x, y = _trace(shifts, [p['n'] for p in info], lo, hi, FWHM[nuc])
    fig, (a0, a1) = plt.subplots(2, 1, figsize=(11, 9.5), gridspec_kw=dict(height_ratios=[1.15, 1], hspace=0.12))
    a0.imshow(img); a0.axis('off')
    for a, pt in pos.items():                                       # 原子番号: 大きく、原子の右上に、ピークと同じ色の丸で
        c = col.get(a, '0.55')
        a0.text(pt.x + 26, pt.y - 26, str(a), color='white', fontsize=13, fontweight='bold', ha='center', va='center',
                bbox=dict(boxstyle='circle,pad=0.25', fc=c, ec='none'), zorder=5)
    a1.plot(x, y, color='0.55', lw=0.8)
    ymax = y.max()
    # 近いピークのラベルが重ならないよう、段をずらす (段の高さは pt で固定。引出線でピークとつなぐ)
    order = sorted(info, key=lambda p: -p['shift'])
    NT = 8
    lastx, used, put = [None] * NT, 0, []
    mind = (hi - lo) * 0.07
    for p in order:
        t = next((t for t in range(NT) if lastx[t] is None or abs(lastx[t] - p['shift']) > mind), NT - 1)
        lastx[t] = p['shift']
        used = max(used, t + 1)
        put.append((p, t))
    a1.set_xlim(hi, lo); a1.set_ylim(0, ymax * 1.06 / (1 - (14 + 34 * used) / 330))
    for p, t in put:
        a1.vlines(p['shift'], 0, ymax * 1.04, color=p['color'], lw=2.4)
        a1.annotate(f"{','.join(map(str, p['atoms']))}\n{p['shift']:.2f}", (p['shift'], ymax * 1.04), xytext=(0, 8 + 34 * t),
                    textcoords='offset points', color=p['color'], ha='center', va='bottom', fontsize=12, fontweight='bold',
                    arrowprops=dict(arrowstyle='-', color=p['color'], lw=0.6, ls=':', shrinkA=0, shrinkB=0))
    a1.set_yticks([])
    for sp in ('top', 'right', 'left'):
        a1.spines[sp].set_visible(False)
    a1.set_xlabel(f'δ {nuc} (ppm)')
    a0.set_title(f'{title}   {nuc}: 原子番号 (RDKit index) と、対応するピークの番号' + ('   (1H は H が結合している原子の番号)' if nuc == '1H' else '') +
                 ('   O-H/N-H は参考値' if any(p['ex'] for p in info) else ''), fontsize=10)
    fig.savefig(path, dpi=130, bbox_inches='tight')
    plt.close(fig)


def write_html(mol0, peaks, nuc, path, title, mhz=None):
    molh, info = _prepare(mol0, peaks, nuc)
    W, H = 640, 420
    dr = rdMolDraw2D.MolDraw2DSVG(W, H)
    dr.drawOptions().addAtomIndices = True
    dr.drawOptions().clearBackground = False
    rdMolDraw2D.PrepareAndDrawMolecule(dr, molh)
    dr.FinishDrawing()
    svg = dr.GetDrawingText()
    svg = svg[svg.index('<svg'):]
    atoms = {}
    for p in info:
        for a in p['atoms']:
            pt = dr.GetDrawCoords(a)
            atoms[a] = dict(x=round(pt.x, 1), y=round(pt.y, 1), peak=p['k'], sym=molh.GetAtomWithIdx(a).GetSymbol())

    shifts = [p['shift'] for p in info]
    span = max(shifts) - min(shifts)
    pad = max(0.08 * span, 0.5 if nuc == '1H' else 8)
    lo, hi = min(shifts) - pad, max(shifts) + pad
    if nuc == '1H':
        lo, hi = min(lo, -0.3), max(hi, 10.3)
    SW, SH, ML, MR, MT, MB = 900, 280, 20, 20, 20, 40
    xp = lambda s: ML + (hi - s) / (hi - lo) * (SW - ML - MR)
    x, y = _trace(shifts, [p['n'] for p in info], lo, hi, FWHM[nuc], 1500)
    ymax = float(y.max())
    base = SH - MB
    yp = lambda v: base - v / ymax * (SH - MB - MT - 30)
    trace = 'M' + ' L'.join(f'{xp(a):.1f},{yp(b):.1f}' for a, b in zip(x, y))
    ticks = ''
    step = 1 if nuc == '1H' else (20 if hi - lo > 100 else 10)
    t = int(np.floor(lo / step)) * step
    while t <= hi:
        if lo <= t <= hi:
            ticks += f'<line x1="{xp(t):.1f}" x2="{xp(t):.1f}" y1="{base}" y2="{base + 5}" stroke="currentColor"/>' \
                     f'<text x="{xp(t):.1f}" y="{base + 18}" text-anchor="middle" font-size="11" fill="currentColor">{t}</text>'
        t += step
    bars = ''
    for p in info:
        px = xp(p['shift'])
        top = yp(p['n'] / max(q['n'] for q in info) * ymax * 0.9) - 12
        bars += (f'<g class="pk" data-k="{p["k"]}" style="--c:{p["color"]}">'
                 f'<rect class="hit" x="{px - 5:.1f}" y="{MT}" width="10" height="{base - MT}" fill="transparent"/>'
                 f'<line class="bar" x1="{px:.1f}" x2="{px:.1f}" y1="{base}" y2="{top:.1f}"/>'
                 f'<text class="lab" x="{px:.1f}" y="{top - 4:.1f}" text-anchor="middle" font-size="10">{html.escape(f"{p["shift"]:.2f}")}</text></g>')
    rows = ''.join(f'<tr class="row" data-k="{p["k"]}" style="--c:{p["color"]}"><td><span class="sw"></span></td>'
                   f'<td>{p["shift"]:.2f}</td><td>{p["n"]}</td><td>{html.escape(p["label"])}</td></tr>' for p in info)
    circles = ''.join(f'<circle class="at" data-a="{a}" data-k="{d["peak"]}" cx="{d["x"]}" cy="{d["y"]}" r="16"/>'
                      for a, d in atoms.items())
    page = f'''<!doctype html><html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(title)} {nuc} 帰属</title><style>
:root{{--bg:#fff;--fg:#1c1c1e;--mut:#6b6b70;--line:#d0d0d5}}
@media(prefers-color-scheme:dark){{:root{{--bg:#16161a;--fg:#ececf0;--mut:#9a9aa2;--line:#3a3a42}}}}
body{{margin:0;padding:16px;background:var(--bg);color:var(--fg);font:14px/1.5 system-ui,"Yu Gothic",sans-serif}}
h1{{font-size:16px;margin:0 0 4px}} .sub{{color:var(--mut);margin:0 0 12px;font-size:13px}}
.wrap{{display:flex;flex-wrap:wrap;gap:16px;align-items:flex-start}}
.mol{{position:relative;width:{W}px;max-width:100%}} .mol svg{{width:100%;height:auto;color:var(--fg)}}
.mol svg path{{stroke:currentColor}} .mol svg text{{fill:currentColor}} .mol svg path[class^=atom]{{fill:currentColor}}
.ov{{position:absolute;left:0;top:0;width:100%;height:100%}}
.at{{fill:transparent;cursor:pointer}} .at.on{{fill:var(--c);fill-opacity:.35;stroke:var(--c);stroke-width:2.5}}
table{{border-collapse:collapse;min-width:260px}} td{{padding:3px 10px;border-bottom:1px solid var(--line)}}
.row{{cursor:pointer}} .row.on{{background:color-mix(in srgb,var(--c) 22%,transparent);font-weight:600}}
.sw{{display:inline-block;width:12px;height:12px;border-radius:3px;background:var(--c)}}
.spec{{margin-top:12px;max-width:100%;overflow-x:auto}} .spec svg{{color:var(--fg);max-width:100%;height:auto}}
.pk{{cursor:pointer}} .pk .bar{{stroke:var(--c);stroke-width:2;opacity:.55}} .pk .lab{{fill:var(--mut)}}
.pk.on .bar{{stroke-width:5;opacity:1;filter:drop-shadow(0 0 5px var(--c))}} .pk.on .lab{{fill:var(--c);font-weight:700;font-size:12px}}
.dim .pk:not(.on) .bar{{opacity:.15}} .hint{{color:var(--mut);font-size:12px;margin-top:6px}}
</style></head><body>
<h1>{html.escape(title)} — {nuc}</h1>
<p class="sub">原子 (丸) か、ピーク (棒・表の行) をクリック。番号は RDKit の原子 index (csv の atom_indices と同じ)。{'1H は H が結合している原子の番号。* は O-H/N-H (参考値)。' if nuc == '1H' else ''}</p>
<div class="wrap"><div class="mol">{svg}<svg class="ov" viewBox="0 0 {W} {H}" id="ov">{circles}</svg></div>
<table><thead><tr><td></td><td>δ (ppm)</td><td>原子数</td><td>{'結合先の原子' if nuc == '1H' else '原子'}</td></tr></thead><tbody>{rows}</tbody></table></div>
<div class="spec"><svg viewBox="0 0 {SW} {SH}" width="{SW}" id="sp"><path d="{trace}" fill="none" stroke="currentColor" stroke-opacity=".35" stroke-width="1"/>
<line x1="{ML}" x2="{SW - MR}" y1="{base}" y2="{base}" stroke="currentColor"/>{ticks}{bars}
<text x="{SW / 2}" y="{SH - 4}" text-anchor="middle" font-size="11" fill="currentColor">δ {nuc} (ppm)</text></svg></div>
<p class="hint">もう一度クリックで解除。表示用に線幅を太くしてあり、多重度は表示していません。</p>
<script>
const colors={json.dumps({p['k']: p['color'] for p in info})};
let cur=null;
const sel=(k)=>{{cur=(cur===k)?null:k;
 document.querySelectorAll('.at').forEach(e=>{{const on=+e.dataset.k===cur;e.classList.toggle('on',on);if(on)e.style.setProperty('--c',colors[cur]);}});
 document.querySelectorAll('.pk,.row').forEach(e=>e.classList.toggle('on',+e.dataset.k===cur));
 document.getElementById('sp').classList.toggle('dim',cur!==null);}};
document.querySelectorAll('.at,.pk,.row').forEach(e=>e.addEventListener('click',()=>sel(+e.dataset.k)));
</script></body></html>'''
    with open(path, 'w', encoding='utf-8') as f:
        f.write(page)


def make_assignment(mol0, peaks, nuc, base, title, mhz=None):
    """<base>_assign.png と <base>_assign.html を出力する。失敗しても本体の処理を止めない。戻り値は出力したパスのリスト。"""
    if not peaks:
        return []
    out = []
    for fn, ext in ((write_png, 'png'), (write_html, 'html')):
        try:
            fn(mol0, peaks, nuc, f'{base}_assign.{ext}', title, mhz)
            out.append(f'{base}_assign.{ext}')
        except Exception as e:                                      # 補助の出力なので、失敗してもスペクトル本体は残す
            print(f'  (帰属図 {ext} は作れませんでした: {e})')
    return out
