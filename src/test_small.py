"""Teszt képeken (data/test_images_small.zip) futtatott kiértékelés + HTML riport.

A test képekhez nincs címke, ezért a riport a predikciók eloszlásáról, a modell
magabiztosságáról, a két fej konzisztenciájáról és a sebességről szól. Ha van
címkéket tartalmazó CSV (--labels: image_path, class_id, MIVS), akkor pontosság,
macro F1 és a mérgező fajok visszahívása is bekerül.

Használat (a repo gyökeréből):
    python src/test_small.py --model models/bioclip_model_fold_1.pth --fold 1   # címkézett mérés
    python src/test_small.py --model models/valami.keras --scale raw            # címkétlen test zip
    python src/test_small.py --model models/snake_multitask_model.h5 --scale unit --limit 500
"""
import argparse
import base64
import io
import json
import os
import time
import zipfile
from pathlib import Path

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
PAD_COLOR = (127, 127, 127)
IMG_EXT = ('.jpg', '.jpeg', '.png')


def letterbox(img, size):
    # Ugyanaz, mint a letterboxing.py: RGB, képarány megtartása, szürke kipárnázás
    img = img.convert('RGB')
    w, h = img.size
    scale = size / max(w, h)
    nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
    img = img.resize((nw, nh), Image.BILINEAR)
    canvas = Image.new('RGB', (size, size), PAD_COLOR)
    canvas.paste(img, ((size - nw) // 2, (size - nh) // 2))
    return np.asarray(canvas, dtype=np.uint8)


def list_images(source):
    if source.is_file():
        with zipfile.ZipFile(source) as z:
            return sorted(n for n in z.namelist() if n.lower().endswith(IMG_EXT))
    return sorted(str(p.relative_to(source)) for p in source.rglob('*') if p.suffix.lower() in IMG_EXT)


def batches(source, names, size, batch_size):
    z = zipfile.ZipFile(source) if source.is_file() else None
    for i in range(0, len(names), batch_size):
        chunk = names[i:i + batch_size]
        arr = np.empty((len(chunk), size, size, 3), dtype=np.uint8)
        for j, n in enumerate(chunk):
            f = io.BytesIO(z.read(n)) if z else source / n
            with Image.open(f) as im:
                arr[j] = letterbox(im, size)
        yield arr
    if z:
        z.close()


def fig_to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=110, bbox_inches='tight')
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def entropy(p):
    return -(p * np.log(np.clip(p, 1e-12, 1))).sum(axis=1)


def macro_f1(y_true, y_pred, n):
    f1 = []
    for c in range(n):
        tp = np.sum((y_true == c) & (y_pred == c))
        fp = np.sum((y_true != c) & (y_pred == c))
        fn = np.sum((y_true == c) & (y_pred != c))
        if tp + fp + fn:
            f1.append(2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True, help='teljes mentett Keras modell (.keras / .h5), vagy BioCLIP state_dict (.pth)')
    ap.add_argument('--arch', choices=['keras', 'bioclip'], default=None,
                    help='alapból a kiterjesztésből: .pth -> bioclip, egyébként keras')
    ap.add_argument('--images', default=str(DATA_DIR / 'test_images_small.zip'), help='zip vagy mappa')
    ap.add_argument('--scale', choices=['raw', 'unit'], default='raw',
                    help='raw: 0-255 (ConvNeXt, uint8 bemenet); unit: /255 (régi ResNet prototípus)')
    ap.add_argument('--fold', type=int, default=None,
                    help='N. fold validációs halmaza (címkékkel!) a data/train_images_small képeiből; a címkétlen test zip helyett')
    ap.add_argument('--labels', default=None, help='opcionális CSV: image_path, class_id, MIVS (EREDETI class_id-kkel)')
    ap.add_argument('--image-dir', default=None, help='--fold esetén a képek mappája (alap: data/train_images_small); pl. data/train_images_medium')
    ap.add_argument('--folds-dir', default=str(DATA_DIR / 'folds'), help='a fold CSV-k mappája (csoportos: data/folds_grouped)')
    ap.add_argument('--sample', type=int, default=None, help='--fold esetén véletlen (seed 42) minta N képpel; lassú gépen a teljes helyett')
    ap.add_argument('--size', type=int, default=None, help='bemeneti méret BioCLIP-hez (alap: a súlyfájlból, 384 ha a nevében van)')
    ap.add_argument('--batch-size', type=int, default=64)
    ap.add_argument('--limit', type=int, default=None, help='csak az első N kép (gyors próba)')
    ap.add_argument('--out', default=str(PROJECT_ROOT / 'plots' / 'test_riport'))
    args = ap.parse_args()

    arch = args.arch or ('bioclip' if args.model.endswith(('.pth', '.pt')) else 'keras')
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model_name = Path(args.model).stem

    mapping = pd.read_csv(DATA_DIR / 'class_id_mapping.csv')
    new2orig = dict(zip(mapping.new_model_id, mapping.original_class_id))
    meta = pd.read_csv(DATA_DIR / 'final_merged_metadata_with_new_labels.csv')
    species = meta.drop_duplicates('class_id').set_index('class_id').sort_index()
    names_sp = species['binomial_name'].to_dict()
    v_map = species['MIVS'].astype(int).to_dict()
    n_cls = len(species)
    train_counts = meta['class_id'].value_counts()

    print(f"Modell betöltése: {args.model} ({arch})")
    if arch == 'bioclip':
        import torch
        import torch.nn as nn
        import open_clip
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        size = args.size or (384 if '384' in Path(args.model).name else 224)

        class BioCLIPMultiTask(nn.Module):  # azonos a model_bioclip.py-val
            def __init__(self, num_classes=296):
                super().__init__()
                m, _, _ = open_clip.create_model_and_transforms('hf-hub:imageomics/bioclip', force_image_size=size)
                self.backbone = m.visual
                self.dropout = nn.Dropout(0.3)
                self.species_head = nn.Linear(self.backbone.output_dim, num_classes)
                self.venomous_head = nn.Linear(self.backbone.output_dim, 1)

            def forward(self, x):
                f = self.dropout(self.backbone(x))
                return self.species_head(f), self.venomous_head(f)

        model = BioCLIPMultiTask(n_cls)
        state = torch.load(args.model, map_location=device)
        if isinstance(state, dict) and 'model' in state:  # resume checkpoint (last_state)
            print(f"Checkpoint: {state.get('epoch')}. epoch, best_val_loss={state.get('best_val_loss')}")
            state = state['model']
        model.load_state_dict(state)
        model.to(device).eval()
        mean = torch.tensor((0.48145466, 0.4578275, 0.40821073), device=device).view(1, 3, 1, 1)
        std = torch.tensor((0.26862954, 0.26130258, 0.27577711), device=device).view(1, 3, 1, 1)

        def run(x):
            with torch.no_grad():
                t = torch.from_numpy(x).to(device).permute(0, 3, 1, 2).float() / 255.0
                s_logit, v_logit = model((t - mean) / std)
                return torch.softmax(s_logit, 1).cpu().numpy(), torch.sigmoid(v_logit).cpu().numpy()
        print(f"Eszköz: {device}")
    else:
        import tensorflow as tf
        model = tf.keras.models.load_model(args.model, compile=False, safe_mode=False)
        size = int(model.input_shape[1])

        def run(x):
            x = x if args.scale == 'raw' else x.astype(np.float32) / 255.0
            return model.predict(x, verbose=0)
    print(f"Bemeneti méret: {size}x{size}")

    fold_df = None
    if args.fold:
        fold_df = pd.read_csv(Path(args.folds_dir) / f'fold_{args.fold}_val.csv')
        source = Path(args.image_dir) if args.image_dir else DATA_DIR / 'train_images_small' / 'train_images_small'
        if source.is_dir() and not any(source.glob('*.*')) and (source / source.name).is_dir():
            source = source / source.name  # zip beágyazott mappája (pl. train_images_medium/train_images_medium)
        if args.sample:
            fold_df = fold_df.sample(n=min(args.sample, len(fold_df)), random_state=42).reset_index(drop=True)
        names = [str(x) for x in fold_df['image_path']]
        args.out = args.out if args.out != str(PROJECT_ROOT / 'plots' / 'test_riport') else str(PROJECT_ROOT / 'plots' / f'fold{args.fold}_riport')
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        model_name = f'{model_name}_fold{args.fold}' + (f'_{Path(args.image_dir).name}' if args.image_dir else '')
    else:
        source = Path(args.images)
        names = list_images(source)
    if args.limit:
        names = names[:args.limit]
    print(f"{len(names)} kép feldolgozása...")

    # bemelegítés, hogy az első batch ne torzítsa a sebességmérést
    run(np.zeros((1, size, size, 3), np.uint8))

    sp_chunks, v_chunks = [], []
    t_load = t_pred = 0.0
    it = batches(source, names, size, args.batch_size)
    while True:
        t0 = time.perf_counter()
        try:
            x = next(it)
        except StopIteration:
            break
        t1 = time.perf_counter()
        s, v = run(x)
        t2 = time.perf_counter()
        t_load += t1 - t0
        t_pred += t2 - t1
        sp_chunks.append(np.asarray(s))
        v_chunks.append(np.asarray(v).reshape(-1))
        print(f"\r  {sum(len(c) for c in sp_chunks)}/{len(names)}", end='', flush=True)
    print()

    probs = np.concatenate(sp_chunks)
    v_prob = np.concatenate(v_chunks)
    n = len(probs)
    top1 = probs.argmax(1)
    conf = probs.max(1)
    top2 = np.sort(probs, axis=1)[:, -2]
    top5_idx = np.argsort(-probs, axis=1)[:, :5]
    ent = entropy(probs)
    v_pred = (v_prob > 0.5).astype(int)
    v_from_species = np.array([v_map[c] for c in top1])
    agree = float(np.mean(v_pred == v_from_species))
    # a fajból származtatott mérgező valószínűség a species_head alapján
    v_sp_prob = probs @ np.array([v_map[c] for c in range(n_cls)], dtype=np.float32)

    # ---- predikciós fájl (a prediction_file_example.csv formátumában) ----
    short = [Path(x).name for x in names]
    pred_df = pd.DataFrame({
        'image_path': short,
        'class_id': [new2orig[c] for c in top1],
        'venomous': v_pred,
    })
    pred_df.to_csv(out / f'{model_name}_predictions.csv', index=False, encoding='utf-8-sig')
    detail = pred_df.assign(species=[names_sp[c] for c in top1], species_conf=conf.round(4),
                            venomous_prob=v_prob.round(4), species_venomous=v_from_species)
    detail.to_csv(out / f'{model_name}_predictions_detail.csv', index=False, encoding='utf-8-sig')

    # ---- metrikák ----
    stats = {
        'kepek_szama': n,
        'modell': model_name,
        'bemeneti_meret': size,
        'osszes_ido_s': round(t_load + t_pred, 2),
        'kepbetoltes_ido_s': round(t_load, 2),
        'modell_ido_s': round(t_pred, 2),
        'modell_ms_per_kep': round(1000 * t_pred / n, 2),
        'modell_kep_per_s': round(n / t_pred, 1),
        'atlag_faj_magabiztossag': round(float(conf.mean()), 4),
        'median_faj_magabiztossag': round(float(np.median(conf)), 4),
        'magabiztos_kepek_90_szazalek': round(float(np.mean(conf >= 0.9)), 4),
        'bizonytalan_kepek_50_alatt': round(float(np.mean(conf < 0.5)), 4),
        'atlag_top1_top2_kulonbseg': round(float((conf - top2).mean()), 4),
        'atlag_entropia': round(float(ent.mean()), 3),
        'megjosolt_fajok_szama': int(len(np.unique(top1))),
        'osszes_faj': n_cls,
        'merges_arany_venomous_head': round(float(v_pred.mean()), 4),
        'merges_arany_fajbol': round(float(v_from_species.mean()), 4),
        'merges_arany_tanito_halmaz': round(float(meta['MIVS'].mean()), 4),
        'fejek_egyezese': round(agree, 4),
        'bizonytalan_merges_zona_0.3_0.7': round(float(np.mean((v_prob > 0.3) & (v_prob < 0.7))), 4),
    }

    has_labels = bool(args.labels or args.fold)
    if has_labels:
        if args.fold:
            lab = fold_df.set_index('image_path').reindex(short)  # már az új (0..295) class_id-k
            ok = lab['class_id'].notna().values
            y = lab.loc[ok, 'class_id'].values.astype(int)
        else:
            lab = pd.read_csv(args.labels).set_index('image_path').reindex(short)
            orig2new = {v: k for k, v in new2orig.items()}
            ok = lab['class_id'].notna().values
            y = lab.loc[ok, 'class_id'].map(orig2new).values.astype(int)
        yv = lab.loc[ok, 'MIVS'].values.astype(int)
        stats['cimkezett_kepek'] = int(ok.sum())
        stats['faj_pontossag'] = round(float(np.mean(top1[ok] == y)), 4)
        stats['faj_top5_pontossag'] = round(float(np.mean([y[i] in top5_idx[ok][i] for i in range(len(y))])), 4)
        stats['faj_macro_f1'] = round(macro_f1(y, top1[ok], n_cls), 4)
        stats['merges_pontossag'] = round(float(np.mean(v_pred[ok] == yv)), 4)
        stats['merges_visszahivas'] = round(float(np.mean(v_pred[ok][yv == 1] == 1)), 4)
        stats['merges_pontossag_fajbol'] = round(float(np.mean(v_from_species[ok] == yv)), 4)
        stats['merges_kigyo_nem_merges_fajkent'] = round(
            float(np.mean(v_from_species[ok][yv == 1] == 0)), 4)

    with open(out / f'{model_name}_stats.json', 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    # ---- ábrák ----
    figs = {}

    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.hist(conf, bins=40, color='#2a7f62')
    ax.axvline(np.median(conf), color='#c0392b', ls='--', label=f'median {np.median(conf):.2f}')
    ax.set(title='Fajpredikció magabiztossága (top-1 valószínűség)', xlabel='valószínűség', ylabel='képek')
    ax.legend()
    figs['conf'] = fig_to_b64(fig)

    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.hist(v_prob, bins=40, color='#b9770e')
    ax.set(title='Mérgezőség-fej kimenete', xlabel='P(mérgező)', ylabel='képek')
    ax.set_yscale('log')
    figs['venom'] = fig_to_b64(fig)

    top = pd.Series(top1).value_counts().head(20)
    fig, ax = plt.subplots(figsize=(7, 6))
    colors = ['#c0392b' if v_map[c] else '#2a7f62' for c in top.index]
    ax.barh([names_sp[c] for c in top.index][::-1], top.values[::-1], color=colors[::-1])
    ax.set(title='20 leggyakrabban megjósolt faj (piros = mérgező)', xlabel='képek')
    figs['top'] = fig_to_b64(fig)

    # megjósolt darabszám vs. tanító gyakoriság (a tesztkészlet eloszlása hasonlítson a tanítóéhoz)
    pc = pd.Series(top1).value_counts().reindex(range(n_cls), fill_value=0)
    tc = train_counts.reindex(range(n_cls), fill_value=0)
    corr = float(np.corrcoef(np.log1p(pc.values), np.log1p(tc.values))[0, 1])
    stats['predikcio_vs_tanito_gyakorisag_korrelacio'] = round(corr, 3)
    fig, ax = plt.subplots(figsize=(5.4, 4.6))
    ax.scatter(tc.values + 1, pc.values + 1, s=14, alpha=.6, color='#34495e')
    ax.set(xscale='log', yscale='log', xlabel='tanító képek száma (+1)', ylabel='megjósolt képek száma (+1)',
           title=f'Predikció vs. tanító gyakoriság (r={corr:.2f}, log)')
    figs['freq'] = fig_to_b64(fig)

    # bizonytalan, ill. legbizonytalanabb mérgezőségi döntések fajonként
    low = pd.DataFrame({'faj': [names_sp[c] for c in top1], 'conf': conf})
    by_sp = low.groupby('faj')['conf'].agg(['mean', 'count']).query('count >= 5').sort_values('mean').head(10)
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.barh(by_sp.index[::-1], by_sp['mean'].values[::-1], color='#8e44ad')
    ax.set(title='Legbizonytalanabb fajok (átl. magabiztosság, min. 5 kép)', xlim=(0, 1))
    figs['lowsp'] = fig_to_b64(fig)

    # ---- minta képek: legbizonytalanabbak ----
    def thumb(idx):
        z = zipfile.ZipFile(source) if source.is_file() else None
        f = io.BytesIO(z.read(names[idx])) if z else source / names[idx]
        with Image.open(f) as im:
            im = im.convert('RGB')
            im.thumbnail((200, 200))
            buf = io.BytesIO()
            im.save(buf, format='JPEG', quality=80)
        if z:
            z.close()
        return base64.b64encode(buf.getvalue()).decode()

    def cards(idx_list, title):
        h = f'<h3>{title}</h3><div class="cards">'
        for i in idx_list:
            h += (f'<div class="card"><img src="data:image/jpeg;base64,{thumb(i)}">'
                  f'<b>{names_sp[top1[i]]}</b><br>{conf[i]*100:.0f}% · '
                  f'{"mérgező" if v_pred[i] else "nem mérgező"} ({v_prob[i]*100:.0f}%)</div>')
        return h + '</div>'

    unsure = np.argsort(conf)[:8]
    sure = np.argsort(-conf)[:8]
    disagree = np.where(v_pred != v_from_species)[0]
    disagree = disagree[np.argsort(-np.abs(v_prob[disagree] - v_sp_prob[disagree]))][:8]
    gallery = cards(unsure, 'Legbizonytalanabb predikciók') + cards(sure, 'Legmagabiztosabb predikciók')
    if len(disagree):
        gallery += cards(disagree, 'A két fej ellentmond egymásnak')

    # ---- HTML ----
    def kpi(label, value, note=''):
        return f'<div class="kpi"><span>{label}</span><b>{value}</b><small>{note}</small></div>'

    kpis = [
        kpi('Képek', f'{n:,}'.replace(',', ' '), f'{size}×{size} letterbox'),
        kpi('Sebesség', f"{stats['modell_kep_per_s']} kép/s", f"{stats['modell_ms_per_kep']} ms/kép (modell)"),
        kpi('Medián magabiztosság', f"{np.median(conf)*100:.1f}%", f'{stats["magabiztos_kepek_90_szazalek"]*100:.0f}% ≥ 90%'),
        kpi('Bizonytalan (<50%)', f'{stats["bizonytalan_kepek_50_alatt"]*100:.1f}%'),
        kpi('Mérgező arány', f'{v_pred.mean()*100:.1f}%', f'tanítóban {meta["MIVS"].mean()*100:.1f}%'),
        kpi('Fejek egyezése', f'{agree*100:.1f}%', 'venomous_head vs. faj → mérgezőség'),
        kpi('Megjósolt fajok', f'{stats["megjosolt_fajok_szama"]} / {n_cls}'),
    ]
    if has_labels:
        kpis += [kpi('Faj pontosság', f'{stats["faj_pontossag"]*100:.1f}%', f'top-5: {stats["faj_top5_pontossag"]*100:.1f}%'),
                 kpi('Macro F1', f'{stats["faj_macro_f1"]:.3f}'),
                 kpi('Mérgező visszahívás', f'{stats["merges_visszahivas"]*100:.1f}%')]
    note = ('' if has_labels else
            '<p class="note">A test képekhez nincs címke, ezért pontosság nem számolható. A riport a predikciók '
            'eloszlását, a magabiztosságot és a sebességet mutatja. Címkével (<code>--labels</code>) '
            'pontosság, macro F1 és visszahívás is bekerül.</p>')

    html = f"""<!doctype html><html lang="hu"><head><meta charset="utf-8">
<title>Teszt riport – {model_name}</title><style>
body{{font-family:system-ui,Segoe UI,sans-serif;margin:0;background:#f5f6f7;color:#1c2a33}}
main{{max-width:1100px;margin:auto;padding:24px 16px}}
h1{{margin:0 0 4px}} h2{{margin-top:34px;border-bottom:2px solid #dde;padding-bottom:4px}}
.sub{{color:#667}} .note{{background:#fff6dd;border-left:4px solid #d9a400;padding:10px 14px}}
.kpis{{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px;margin:18px 0}}
.kpi{{background:#fff;border-radius:8px;padding:12px 14px;box-shadow:0 1px 3px #0001}}
.kpi span{{display:block;color:#667;font-size:12px}} .kpi b{{font-size:22px}} .kpi small{{display:block;color:#889}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(380px,1fr));gap:16px}}
.grid img{{width:100%;background:#fff;border-radius:8px;box-shadow:0 1px 3px #0001}}
.cards{{display:flex;flex-wrap:wrap;gap:10px}}
.card{{background:#fff;border-radius:8px;padding:8px;width:200px;font-size:12px;box-shadow:0 1px 3px #0001}}
.card img{{width:100%;border-radius:4px}}
table{{border-collapse:collapse;background:#fff}} td,th{{padding:5px 12px;border-bottom:1px solid #eee;text-align:left}}
</style></head><body><main>
<h1>Modell teljesítmény – teszt képek</h1>
<div class="sub">{model_name} · {n} kép · {time.strftime('%Y-%m-%d %H:%M')}</div>
{note}
<div class="kpis">{''.join(kpis)}</div>
<h2>Magabiztosság</h2><div class="grid">
<img src="data:image/png;base64,{figs['conf']}"><img src="data:image/png;base64,{figs['venom']}"></div>
<h2>Predikciók eloszlása</h2><div class="grid">
<img src="data:image/png;base64,{figs['top']}"><img src="data:image/png;base64,{figs['freq']}">
<img src="data:image/png;base64,{figs['lowsp']}"></div>
<h2>Minták</h2>{gallery}
<h2>Részletes számok</h2><table>{''.join(f'<tr><th>{k}</th><td>{v}</td></tr>' for k, v in stats.items())}</table>
</main></body></html>"""
    (out / f'{model_name}_riport.html').write_text(html, encoding='utf-8')

    print(f"\nKész. Riport: {out / (model_name + '_riport.html')}")
    print(f"Predikciók: {out / (model_name + '_predictions.csv')}")
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
