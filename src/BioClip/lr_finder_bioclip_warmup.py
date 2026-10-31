# -*- coding: utf-8 -*-
"""
LR finder a BEMELEGÍTÉSI fázishoz (BioCLIP, fagyasztott backbone, csak a fejek tanulnak).
Exponenciális LR-sweep a fold_1 tanító halmazon, EMA-simítással; az ajánlott érték a
legmeredekebb esés és a loss-minimum / MIN_DIV közül a kisebb, a futások mértani átlaga.
A kiírt WARMUP_LR értéket írd be a model_bioclip.py WARMUP_LR változójába.
"""
import os
import json
import shutil
import zipfile
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import open_clip

SEED = 42

# ---------------------------------------------------------
# 1. KÖRNYEZET ÉS ADATÚTVONALAK
# ---------------------------------------------------------
# Csoportos (observation_id szerinti) foldok és ugyanaz az augmentáció, mint a model_bioclip.py-ban.
# A kimenet (log/JSON, a backbone finderhez a bemelegítési súlyok) a model_bioclip.py mappáit használja.
FOLDS_NAME = 'folds_grouped'
AUGMENTATION = 'strong'   # 'basic' vagy 'strong'; egyezzen a model_bioclip.py-ban beállítottal
_TAG = '_aug' if AUGMENTATION == 'strong' else ''
MODELS_SUBDIR = 'bioclip_grouped' + _TAG
LOGS_SUBDIR = 'BioClip_grouped' + _TAG

IMAGE_FOLDER_NAME = "train_images_224_letterboxed"
CSV_NAME = "fold_1_train.csv"

IN_COLAB = True
try:
    from google.colab import drive
    drive.mount('/content/drive')
except Exception:
    IN_COLAB = False

if IN_COLAB:
    DRIVE_DIR = Path('/content/drive/MyDrive/Mamba')
    ZIP_PATH = DRIVE_DIR / f'Databank/Train data/{IMAGE_FOLDER_NAME}.zip'
    EXTRACT_PATH = Path('/content/Databank/Train data')
    FOLDS_SOURCE_DIR = DRIVE_DIR / 'CSV files' / FOLDS_NAME
    LOGS_DIR = DRIVE_DIR / 'Logs' / LOGS_SUBDIR
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    if ZIP_PATH.exists() and not (EXTRACT_PATH / IMAGE_FOLDER_NAME).exists():
        print("[+] Képek kicsomagolása a helyi SSD-re...")
        EXTRACT_PATH.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
            zip_ref.extractall(EXTRACT_PATH)

    LOCAL_FOLDS_DIR = Path(f'/content/data_local/{FOLDS_NAME}')
    LOCAL_FOLDS_DIR.mkdir(parents=True, exist_ok=True)
    src_csv, dst_csv = FOLDS_SOURCE_DIR / CSV_NAME, LOCAL_FOLDS_DIR / CSV_NAME
    if src_csv.exists() and not dst_csv.exists(): shutil.copy2(src_csv, dst_csv)

    DATA_DIR = LOCAL_FOLDS_DIR if dst_csv.exists() else FOLDS_SOURCE_DIR
    IMAGE_DIR = EXTRACT_PATH / IMAGE_FOLDER_NAME
else:
    matplotlib.use("Agg")
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    DATA_DIR = PROJECT_ROOT / "data" / FOLDS_NAME
    IMAGE_DIR = PROJECT_ROOT / "data" / IMAGE_FOLDER_NAME
    LOGS_DIR = PROJECT_ROOT / "plots"
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

PLOT_PATH = LOGS_DIR / "lr_finder_bioclip_warmup_plot.png"
JSON_PATH = LOGS_DIR / "lr_finder_bioclip_warmup.json"

# ---------------------------------------------------------
# 2. LR FINDER BEÁLLÍTÁSOK
# ---------------------------------------------------------
NUM_CLASSES = 296
BATCH_SIZE = 128          # ugyanaz, mint a model_bioclip.py-ban
SWEEP_STEPS = 200
N_RUNS = 2
START_LR = 1e-6
END_LR = 1e-1
LR_CLIP = (1e-5, 1e-2)
DEFAULT_LR = 1e-3         # ha a sweep nem értékelhető
SMOOTH_BETA = 0.85
DIVERGENCE_FACTOR = 4.0
MIN_DIV = 3.0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

v_mapping_list = [
    0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0,
    1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0,
    0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0
]
V_MAPPING = torch.tensor(v_mapping_list, dtype=torch.float32).to(DEVICE)

# ---------------------------------------------------------
# 3. DATASET ÉS AUGMENTÁCIÓ (azonos a model_bioclip.py-val)
# ---------------------------------------------------------
class SnakeDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.df = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row['image_path'])
        image = Image.open(img_path).convert('RGB')
        if self.transform: image = self.transform(image)
        return image, torch.tensor(row['class_id'], dtype=torch.long), torch.tensor(row['MIVS'], dtype=torch.float32)

normalize = transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
if AUGMENTATION == 'strong':
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.5, 1.0), ratio=(0.75, 1.3333)),
        transforms.RandomHorizontalFlip(), transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02),
        transforms.ToTensor(), normalize,
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.1)),
    ])
else:
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(), transforms.RandomRotation(15), transforms.ToTensor(), normalize
    ])

# ---------------------------------------------------------
# 4. MODELL ÉS KÖLTSÉGFÜGGVÉNY (azonos a model_bioclip.py-val)
# ---------------------------------------------------------
class BioCLIPMultiTask(nn.Module):
    def __init__(self, num_classes=296):
        super().__init__()
        print("[+] BioCLIP Foundation Modell letöltése...")
        model, _, _ = open_clip.create_model_and_transforms('hf-hub:imageomics/bioclip')
        self.backbone = model.visual
        self.dropout = nn.Dropout(0.3)
        self.species_head = nn.Linear(self.backbone.output_dim, num_classes)
        self.venomous_head = nn.Linear(self.backbone.output_dim, 1)

    def forward(self, x):
        f = self.dropout(self.backbone(x))
        return self.species_head(f), self.venomous_head(f)

def compute_loss(logits_species, logits_venom, target_species, target_venom):
    probs = torch.softmax(logits_species, dim=-1)
    v_y = V_MAPPING[target_species]
    v_y_pred = torch.sum(probs * V_MAPPING, dim=-1)

    weight_if_venomous = 5.0 * (1.0 - v_y_pred) + 2.0 * v_y_pred
    weight_if_non_venomous = 1.0 * (1.0 - v_y_pred) + 2.0 * v_y_pred
    weight = v_y * weight_if_venomous + (1.0 - v_y) * weight_if_non_venomous

    ce_loss = nn.functional.cross_entropy(logits_species, target_species, reduction='none')
    species_loss = torch.mean(ce_loss * weight)

    target_venom = target_venom.unsqueeze(1)
    weights = 1.0 + 4.0 * target_venom
    bce_loss = nn.functional.binary_cross_entropy_with_logits(logits_venom, target_venom, reduction='none')
    venom_loss = torch.mean(bce_loss * weights)

    return species_loss + 0.2 * venom_loss

# ---------------------------------------------------------
# 5. LEARNING RATE FINDER KERETRENDSZER
# ---------------------------------------------------------
def ema(values, beta):
    out = np.empty(len(values), dtype=np.float64)
    for i, v in enumerate(values):
        out[i] = v if i == 0 else beta * out[i - 1] + (1.0 - beta) * v
    return out

def moving_average(x, window):
    x = np.asarray(x, dtype=np.float64)
    window = max(1, int(window))
    if window == 1 or len(x) < 2: return x
    pad = window // 2
    xp = np.pad(x, (pad, window - 1 - pad), mode='edge')
    return np.convolve(xp, np.ones(window) / window, mode='valid')

def analyze_sweep(lrs, raw_losses, diverged=False, beta=SMOOTH_BETA, skip_begin=15, min_div=MIN_DIV, clip=LR_CLIP):
    lrs = np.asarray(lrs, dtype=np.float64); raw = np.asarray(raw_losses, dtype=np.float64)
    n = min(len(lrs), len(raw)); lrs, raw = lrs[:n], raw[:n]
    out = {'ok': False, 'n_steps': int(n), 'diverged': bool(diverged), 'warnings': []}
    finite = np.isfinite(raw)
    if not finite.all():
        n = int(np.argmax(~finite)); lrs, raw = lrs[:n], raw[:n]; out['n_steps'] = n
    if n < 40: return out

    sm = ema(raw, beta); lag = int(round(beta / (1.0 - beta)))
    skip = min(skip_begin, n // 4); idx_min = skip + int(np.argmin(sm[skip:]))
    out.update(lag_steps=lag, idx_min=int(idx_min))

    seg = sm[skip:idx_min + 1]; grad = moving_average(np.gradient(seg), 7)
    idx_steep = skip + int(np.argmin(grad))
    idx_steep_corr = max(skip, idx_steep - lag); idx_min_corr = max(0, idx_min - lag)

    lr_steepest = float(lrs[idx_steep_corr]); lr_min_loss = float(lrs[idx_min_corr])
    rec_raw = min(lr_steepest, lr_min_loss / min_div)
    rec = float(np.clip(rec_raw, clip[0], clip[1]))

    if rec != rec_raw:
        out['warnings'].append(f"Az ajánlás a clip határára esett ({rec_raw:.2e} -> {rec:.2e}).")
    if not diverged and idx_min >= n - lag - 5:
        out['warnings'].append("A loss a sweep végéig csökkent: a keresési sáv felső határa túl alacsony lehet.")

    out.update(ok=True, lr_steepest=lr_steepest, lr_min_loss=lr_min_loss, recommended_lr=rec, min_smoothed_loss=float(sm[idx_min]))
    return out

def aggregate_runs(analyses, default_lr, clip=LR_CLIP):
    good = [a for a in analyses if a.get('ok')]
    notes = [w for a in analyses for w in a.get('warnings', [])]
    if not good: return float(default_lr), notes + ["Nincs értékelhető futás, az alapértelmezett LR marad."]
    recs = np.array([a['recommended_lr'] for a in good], dtype=np.float64)
    if len(recs) > 1 and recs.max() / recs.min() > 3.0:
        notes.append(f"A futások ajánlásai jelentősen eltérnek ({recs.min():.2e} .. {recs.max():.2e}).")
    final = float(np.clip(float(np.exp(np.mean(np.log(recs)))), clip[0], clip[1]))
    return final, notes

class LRFinder:
    """Exponenciálisan növeli az LR-t lépésenként és naplózza a (nyers) loss-t."""
    def __init__(self, optimizer, start_lr, end_lr, max_steps, beta=SMOOTH_BETA, divergence_factor=DIVERGENCE_FACTOR):
        self.optimizer = optimizer
        self.start_lr = start_lr; self.max_steps = max_steps
        self.beta = beta; self.divergence_factor = divergence_factor
        self.factor = (end_lr / start_lr) ** (1.0 / max_steps)
        self.lrs = []; self.losses = []; self.smoothed = []
        self.best_smooth = float('inf'); self.step = 0
        self.diverged_at = None

    def set_lr(self):
        lr = self.start_lr * (self.factor ** self.step)
        for g in self.optimizer.param_groups: g['lr'] = lr
        self.lrs.append(lr)

    def record(self, loss):
        """True-t ad vissza, ha a sweep-et le kell állítani (divergencia)."""
        self.losses.append(loss)
        n = self.step
        self.step += 1
        if not np.isfinite(loss):
            self.diverged_at = self.step
            return True
        smooth = loss if n == 0 else self.beta * self.smoothed[-1] + (1.0 - self.beta) * loss
        self.smoothed.append(smooth)
        self.best_smooth = min(self.best_smooth, smooth)
        if self.step > 20 and smooth > self.divergence_factor * self.best_smooth:
            self.diverged_at = self.step
            return True
        return False

def plot_and_save(finders, analyses, final_lr, save_path, title, lr_label):
    plt.figure(figsize=(11, 5.5))
    palette = ['#2b5c8f', '#2e8b57']
    for k, (f, a) in enumerate(zip(finders, analyses)):
        n = min(len(f.lrs), len(f.smoothed), len(f.losses))
        col = palette[k % len(palette)]
        plt.plot(f.lrs[:n], f.losses[:n], color=col, alpha=0.12, linewidth=1)
        plt.plot(f.lrs[:n], f.smoothed[:n], color=col, linewidth=2, label=f'{k + 1}. futás')
        if a.get('ok'):
            plt.axvline(a['lr_min_loss'], color=col, linestyle=':', linewidth=1.2)
            plt.axvline(a['lr_steepest'], color=col, linestyle='-.', linewidth=1.2)
    plt.axvline(final_lr, color='#d9534f', linestyle='--', linewidth=2.5, label=f'{lr_label}: {final_lr:.2e}')
    plt.xscale('log'); plt.xlabel('Tanulási ráta (log skála)'); plt.ylabel('Költségérzékeny loss')
    plt.title(title); plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
    plt.savefig(save_path, dpi=130)
    if IN_COLAB: plt.show()
    plt.close()
    print(f"[LRFinder] Grafikon elmentve ide: {save_path}")

def report_and_save(analyses, final_lr, notes, json_key, constant_name):
    print("\n" + "=" * 60)
    for n in notes: print(f"[!] {n}")
    print(f"--> VÉGSŐ ÉRTÉK: {final_lr:.2e}")
    print(f"--> Ezt ({final_lr:.6f}) írd be a model_bioclip.py {constant_name} változójába!")
    print("=" * 60)
    with open(JSON_PATH, 'w', encoding='utf-8') as fh:
        json.dump({json_key: final_lr, 'runs': analyses, 'notes': notes}, fh, indent=2, ensure_ascii=False)

# ---------------------------------------------------------
# 6. FŐPROGRAM
# ---------------------------------------------------------
def infinite_batches(loader):
    while True:
        for batch in loader: yield batch

def run_sweep(train_ds, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    gen = torch.Generator(); gen.manual_seed(seed)
    loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, num_workers=0,
                        pin_memory=DEVICE.type == "cuda", generator=gen)

    model = BioCLIPMultiTask(NUM_CLASSES).to(DEVICE)
    # Bemelegítés: a backbone fagyasztott, csak a fejek tanulnak (mint a model_bioclip.py 1. fázisában)
    for param in model.backbone.parameters(): param.requires_grad = False
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=START_LR, weight_decay=1e-4)

    finder = LRFinder(optimizer, START_LR, END_LR, SWEEP_STEPS)
    model.train()
    batches = infinite_batches(loader)
    for step in range(SWEEP_STEPS):
        images, species, venom = next(batches)
        images, species, venom = images.to(DEVICE), species.to(DEVICE), venom.to(DEVICE)
        finder.set_lr()
        optimizer.zero_grad()
        s_logits, v_logits = model(images)
        loss = compute_loss(s_logits, v_logits, species, venom)
        loss.backward()
        optimizer.step()
        stop = finder.record(float(loss.item()))
        if (step + 1) % 20 == 0:
            print(f"  lépés {step + 1:3d}/{SWEEP_STEPS} | LR: {finder.lrs[-1]:.2e} | loss: {finder.losses[-1]:.4f}")
        if stop:
            print(f"  [LRFinder] Divergencia a(z) {finder.step}. lépésnél (LR={finder.lrs[-1]:.2e}) -> leállítás.")
            break

    del model, optimizer, loader
    if DEVICE.type == "cuda": torch.cuda.empty_cache()
    return finder

def main():
    print(f"Adatok: {DATA_DIR / CSV_NAME} | Képek: {IMAGE_DIR} | Eszköz: {DEVICE}")
    train_ds = SnakeDataset(DATA_DIR / CSV_NAME, IMAGE_DIR, transform=train_transform)

    finders, analyses = [], []
    for run in range(N_RUNS):
        seed = SEED + run
        print(f"\n=== {run + 1}/{N_RUNS}. futás (Warm-up LR keresés, {START_LR:.0e} -> {END_LR:.0e}) | seed={seed} ===")
        finder = run_sweep(train_ds, seed)
        analysis = analyze_sweep(finder.lrs, finder.losses, diverged=finder.diverged_at is not None)
        finders.append(finder); analyses.append(analysis)

    final_lr, notes = aggregate_runs(analyses, DEFAULT_LR)
    report_and_save(analyses, final_lr, notes, 'warmup_lr', 'WARMUP_LR')
    plot_and_save(finders, analyses, final_lr, PLOT_PATH, 'LR Finder eredménye (BioCLIP - Warm-up)', 'Ajánlott Warm-up LR')

if __name__ == "__main__":
    main()
