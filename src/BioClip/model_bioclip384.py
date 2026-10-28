# -*- coding: utf-8 -*-
"""
BioCLIP finomhangolás 224 -> 384 (Progressive Resizing).
Betölti a kész 224-es finomhangolt modellt (bioclip_model_fold_X.pth), a ViT pozíciós
beágyazását bikubikusan 14x14-ről 24x24 rácsra interpolálja, majd teljes finomhangolást végez
384x384-es képeken. Előfeltétel: a model_bioclip.py lefutott az adott foldra.
Az epoch-onkénti checkpoint miatt a Colab-megszakadás után a futtatás folytatódik.
"""
import os
import json
import shutil
import zipfile
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import open_clip
from sklearn.metrics import f1_score

SEED = 42
torch.manual_seed(SEED); np.random.seed(SEED)

# ---------------------------------------------------------
# 1. KÖRNYEZET ÉS ADATÚTVONALAK
# ---------------------------------------------------------
IMAGE_FOLDER_NAME = "train_images_384_letterboxed"

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
    FOLDS_SOURCE_DIR = DRIVE_DIR / 'CSV files' / 'folds'
    LOGS_DIR = DRIVE_DIR / 'Logs'
    BIOCLIP_MODELS_DIR = DRIVE_DIR / 'models' / 'bioclip'
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    if ZIP_PATH.exists() and not (EXTRACT_PATH / IMAGE_FOLDER_NAME).exists():
        print("[+] Képek kicsomagolása a helyi SSD-re...")
        EXTRACT_PATH.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
            zip_ref.extractall(EXTRACT_PATH)
        print("Kicsomagolás kész!")
    elif not ZIP_PATH.exists():
        print(f"FIGYELEM: A ZIP nem található: {ZIP_PATH}")

    LOCAL_FOLDS_DIR = Path('/content/data_local/folds')
    LOCAL_FOLDS_DIR.mkdir(parents=True, exist_ok=True)
    if FOLDS_SOURCE_DIR.exists():
        for i in range(1, 6):
            for split in ["train", "val"]:
                f = f"fold_{i}_{split}.csv"
                src, dst = FOLDS_SOURCE_DIR / f, LOCAL_FOLDS_DIR / f
                if src.exists() and not dst.exists(): shutil.copy2(src, dst)

    DATA_DIR = LOCAL_FOLDS_DIR if (LOCAL_FOLDS_DIR / "fold_1_train.csv").exists() else FOLDS_SOURCE_DIR
    IMAGE_DIR = EXTRACT_PATH / IMAGE_FOLDER_NAME
else:
    matplotlib.use("Agg")
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    DATA_DIR = PROJECT_ROOT / "data" / "folds"
    IMAGE_DIR = PROJECT_ROOT / "data" / IMAGE_FOLDER_NAME
    LOGS_DIR = PROJECT_ROOT / "plots"
    BIOCLIP_MODELS_DIR = PROJECT_ROOT / "models" / "bioclip"
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------
# 2. HIPERPARAMÉTEREK
# ---------------------------------------------------------
IMAGE_SIZE = 384
NUM_CLASSES = 296
NUM_FOLDS = 5
BATCH_SIZE = 32           # 384-en a ViT 576 tokent dolgoz fel, ezért kisebb batch kell
NUM_WORKERS = 2 if IN_COLAB else 0

FINETUNE_EPOCHS = 20
# A 224-es finomhangolás LR-jénél (lr_finder_bioclip_backbone.py) ne legyen nagyobb:
# a modell már tanult, itt csak az új felbontáshoz igazodik.
FINETUNE_LR = 2e-5
WEIGHT_DECAY = 1e-4

FOLDS_TO_RUN = [1]
EARLY_STOPPING_PATIENCE = 5
EARLY_STOPPING_MIN_DELTA = 1e-3

USE_AMP = True            # vegyes pontosság (bf16, ha támogatott, különben fp16 + GradScaler)
GRAD_CHECKPOINTING = True # kevesebb VRAM, ~25-30%-kal lassabb; A100-on kikapcsolható

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
AMP_ENABLED = USE_AMP and DEVICE.type == "cuda"
AMP_DTYPE = torch.bfloat16 if (AMP_ENABLED and torch.cuda.is_bf16_supported()) else torch.float16

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
V_MAPPING_NP = np.array(v_mapping_list)

# ---------------------------------------------------------
# 3. DATASET ÉS AUGMENTÁCIÓ (azonos a 224-es szkripttel)
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
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(), transforms.RandomRotation(15), transforms.ToTensor(), normalize
])
val_transform = transforms.Compose([transforms.ToTensor(), normalize])

# ---------------------------------------------------------
# 4. MODELL ÉS KÖLTSÉGFÜGGVÉNYEK
# ---------------------------------------------------------
class BioCLIPMultiTask(nn.Module):
    def __init__(self, num_classes=296, image_size=224):
        super().__init__()
        print(f"[+] BioCLIP Foundation Modell letöltése ({image_size}x{image_size})...")
        # force_image_size: az open_clip a betöltéskor a pozíciós beágyazást az új rácsra interpolálja
        model, _, _ = open_clip.create_model_and_transforms('hf-hub:imageomics/bioclip', force_image_size=image_size)
        self.backbone = model.visual
        self.dropout = nn.Dropout(0.3)
        self.species_head = nn.Linear(self.backbone.output_dim, num_classes)
        self.venomous_head = nn.Linear(self.backbone.output_dim, 1)

    def forward(self, x):
        f = self.dropout(self.backbone(x))
        return self.species_head(f), self.venomous_head(f)

def compute_loss(logits_species, logits_venom, target_species, target_venom):
    # A loss mindig fp32-ben számolódik, AMP mellett is
    logits_species, logits_venom = logits_species.float(), logits_venom.float()
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

def interpolate_pos_embed(pos_embed, new_grid):
    """[1 + g*g, C] pozíciós beágyazás bikubikus átméretezése új g' x g' rácsra (a CLS token marad)."""
    tok, img = pos_embed[:1], pos_embed[1:]
    old_grid = int(round(img.shape[0] ** 0.5))
    if old_grid == new_grid: return pos_embed
    img = img.reshape(1, old_grid, old_grid, -1).permute(0, 3, 1, 2).float()
    img = F.interpolate(img, size=(new_grid, new_grid), mode='bicubic', antialias=True, align_corners=False)
    img = img.permute(0, 2, 3, 1).reshape(new_grid * new_grid, -1).to(pos_embed.dtype)
    return torch.cat([tok, img], dim=0)

def load_224_weights(model, path):
    """A 224-es finomhangolt súlyok betöltése a 384-es modellbe (pozíciós beágyazás interpolálásával)."""
    state = torch.load(path, map_location='cpu')
    key = 'backbone.positional_embedding'
    new_grid = model.backbone.grid_size[0]
    state[key] = interpolate_pos_embed(state[key], new_grid)
    expected = model.backbone.positional_embedding.shape
    assert state[key].shape == expected, f"Pozíciós beágyazás mérete nem egyezik: {tuple(state[key].shape)} != {tuple(expected)}"
    model.load_state_dict(state)

# ---------------------------------------------------------
# 5. VALIDÁCIÓS SEGÉDFÜGGVÉNY ÉS ÁBRA
# ---------------------------------------------------------
def evaluate_model(model, val_loader):
    model.eval()
    val_loss = 0.0
    all_preds_s, all_targets_s = [], []
    all_preds_v, all_targets_v = [], []

    with torch.no_grad():
        for images, species, venom in val_loader:
            images, species, venom = images.to(DEVICE), species.to(DEVICE), venom.to(DEVICE)
            with torch.autocast(device_type=DEVICE.type, dtype=AMP_DTYPE, enabled=AMP_ENABLED):
                s_logits, v_logits = model(images)
            loss = compute_loss(s_logits, v_logits, species, venom)
            val_loss += loss.item()

            all_preds_s.append(s_logits.argmax(dim=-1).cpu().numpy())
            all_targets_s.append(species.cpu().numpy())
            all_preds_v.append((torch.sigmoid(v_logits.float()) > 0.5).int().cpu().numpy().flatten())
            all_targets_v.append(venom.int().cpu().numpy())

    y_pred_s = np.concatenate(all_preds_s)
    y_true_s = np.concatenate(all_targets_s)
    y_pred_v = np.concatenate(all_preds_v)
    y_true_v = np.concatenate(all_targets_v)

    v_loss_avg = val_loss / len(val_loader)
    val_acc = (y_pred_s == y_true_s).mean()
    val_macro_f1 = f1_score(y_true_s, y_pred_s, average='macro')
    val_venom_acc = (y_pred_v == y_true_v).mean()

    v_y = V_MAPPING_NP[y_true_s]
    v_y_hat = V_MAPPING_NP[y_pred_s]
    base_penalty = 1.0 + 4.0 * v_y + (1.0 - 4.0 * v_y) * v_y_hat
    val_weighted_penalty = np.mean(base_penalty * (y_true_s != y_pred_s).astype(float))

    return v_loss_avg, val_acc, val_macro_f1, val_venom_acc, val_weighted_penalty

def plot_history(history, save_path, fold):
    plt.figure(figsize=(10, 6))
    plt.plot(history['train_loss'], label="Tanítási Loss", linewidth=2)
    plt.plot(history['val_loss'], label="Validációs Loss", linewidth=2)
    plt.title(f"BioCLIP 384 Költségfüggvény (Fold {fold} - Finomhangolás)")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend(); plt.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(save_path, dpi=150); plt.close()

# ---------------------------------------------------------
# 6. EGY FOLD FINOMHANGOLÁSA 224 -> 384
# ---------------------------------------------------------
def train_fold(fold):
    print("\n" + "=" * 65)
    print(f"=== {fold}. FOLD: BioCLIP 224 -> 384 Progressive Resizing ===")
    print("=" * 65)

    BIOCLIP_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    train_csv = DATA_DIR / f"fold_{fold}_train.csv"
    val_csv = DATA_DIR / f"fold_{fold}_val.csv"

    previous_weights_path = BIOCLIP_MODELS_DIR / f"bioclip_model_fold_{fold}.pth"
    best_weights_path = BIOCLIP_MODELS_DIR / f"bioclip_model_384_fold_{fold}.pth"
    last_state_path = BIOCLIP_MODELS_DIR / f"bioclip_384_last_state_fold_{fold}.pth"

    log_path = LOGS_DIR / f"bioclip_384_finetune_log_fold_{fold}.csv"
    plot_path = LOGS_DIR / f"bioclip_384_plot_fold_{fold}.png"
    result_path = LOGS_DIR / f"bioclip_384_fold_{fold}_result.json"

    for required in [train_csv, val_csv]:
        if not required.exists():
            print(f"[!] HIÁNYZÓ ADATFÁJL: {required}"); return
    if not IMAGE_DIR.exists():
        print(f"[!] A képmappa nem található: {IMAGE_DIR}"); return
    if not previous_weights_path.exists():
        print(f"\n[!] Nem található a 224-es kiinduló modell:\n{previous_weights_path}")
        print("Először futtasd le a model_bioclip.py-t erre a foldra!"); return

    resume = None
    if last_state_path.exists():
        resume = torch.load(last_state_path, map_location='cpu')
        if resume.get('finished'):
            print(f"\n[+] A 384-es modell már elkészült: {best_weights_path}")
            print("Újratanításhoz töröld a bioclip_384_last_state_fold_X.pth fájlt.")
            return

    train_ds = SnakeDataset(train_csv, IMAGE_DIR, transform=train_transform)
    val_ds = SnakeDataset(val_csv, IMAGE_DIR, transform=val_transform)
    pin = DEVICE.type == "cuda"
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, num_workers=NUM_WORKERS,
                              pin_memory=pin, persistent_workers=NUM_WORKERS > 0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS,
                            pin_memory=pin, persistent_workers=NUM_WORKERS > 0)

    model = BioCLIPMultiTask(NUM_CLASSES, image_size=IMAGE_SIZE).to(DEVICE)
    print(f"[+] A 224-es súlyok betöltése (pozíciós beágyazás -> {model.backbone.grid_size[0]}x{model.backbone.grid_size[1]}): {previous_weights_path}")
    load_224_weights(model, previous_weights_path)
    model.backbone.set_grad_checkpointing(GRAD_CHECKPOINTING)
    for param in model.backbone.parameters(): param.requires_grad = True

    optimizer = optim.AdamW(model.parameters(), lr=FINETUNE_LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.3, patience=2, min_lr=1e-6)
    scaler = torch.amp.GradScaler('cuda', enabled=AMP_ENABLED and AMP_DTYPE == torch.float16)

    csv_header = "epoch,train_loss,val_loss,val_acc,val_macro_f1,val_venom_acc,val_weighted_penalty\n"
    best_val_loss = float('inf')
    patience_counter = 0
    start_epoch = 0
    history = {'train_loss': [], 'val_loss': [], 'val_acc': [], 'val_macro_f1': [], 'val_weighted_penalty': []}

    if resume is not None:
        model.load_state_dict(resume['model'])
        optimizer.load_state_dict(resume['optimizer'])
        scheduler.load_state_dict(resume['scheduler'])
        scaler.load_state_dict(resume['scaler'])
        best_val_loss = resume['best_val_loss']; patience_counter = resume['patience_counter']
        start_epoch = resume['epoch']; history = resume['history']
        print(f"[+] Folytatás az utolsó checkpointból: {start_epoch}. epoch kész, best val loss: {best_val_loss:.4f}")
    else:
        with open(log_path, 'w') as f: f.write(csv_header)

    print(f"\n=== Finomhangolás {IMAGE_SIZE}x{IMAGE_SIZE} (batch={BATCH_SIZE}, LR={FINETUNE_LR:.1e}, "
          f"AMP={AMP_DTYPE if AMP_ENABLED else 'ki'}, grad-ckpt={GRAD_CHECKPOINTING}) ===")

    for epoch in range(start_epoch, FINETUNE_EPOCHS):
        model.train()
        train_loss = 0.0
        for images, species, venom in train_loader:
            images, species, venom = images.to(DEVICE), species.to(DEVICE), venom.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=DEVICE.type, dtype=AMP_DTYPE, enabled=AMP_ENABLED):
                s_logits, v_logits = model(images)
            loss = compute_loss(s_logits, v_logits, species, venom)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()

        t_loss_avg = train_loss / len(train_loader)
        v_loss_avg, v_acc, v_f1, v_venom, v_pen = evaluate_model(model, val_loader)

        history['train_loss'].append(t_loss_avg); history['val_loss'].append(v_loss_avg)
        history['val_acc'].append(float(v_acc)); history['val_macro_f1'].append(float(v_f1))
        history['val_weighted_penalty'].append(float(v_pen))

        with open(log_path, 'a') as f:
            f.write(f"{epoch+1},{t_loss_avg:.6f},{v_loss_avg:.6f},{v_acc:.6f},{v_f1:.6f},{v_venom:.6f},{v_pen:.6f}\n")

        print(f"Fine-tune 384 {epoch+1:02d} | Train Loss: {t_loss_avg:.4f} | Val Loss: {v_loss_avg:.4f} | Acc: {v_acc:.4f} | F1: {v_f1:.4f} | Pen: {v_pen:.4f}")

        scheduler.step(v_loss_avg)

        stop = False
        if v_loss_avg < best_val_loss - EARLY_STOPPING_MIN_DELTA:
            best_val_loss = v_loss_avg
            patience_counter = 0
            torch.save(model.state_dict(), best_weights_path)
            print("  [+] Best model mentve!")
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print(f"  [!] Early Stopping aktiválódott a(z) {epoch+1}. epochnál!")
                stop = True

        finished = stop or (epoch + 1 == FINETUNE_EPOCHS)
        torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
                    'scaler': scaler.state_dict(), 'best_val_loss': best_val_loss, 'patience_counter': patience_counter,
                    'epoch': epoch + 1, 'history': history, 'finished': finished}, last_state_path)
        if stop: break

    if not best_weights_path.exists():  # nem javult egyik epoch sem a min_delta-hoz képest
        torch.save(model.state_dict(), best_weights_path)

    plot_history(history, plot_path, fold)
    best_idx = int(np.argmin(history['val_loss']))
    with open(result_path, 'w', encoding='utf-8') as fh:
        json.dump({'fold': fold, 'image_size': IMAGE_SIZE, 'finetune_lr': FINETUNE_LR, 'batch_size': BATCH_SIZE,
                   'epochs_run': len(history['val_loss']), 'best_epoch': best_idx + 1,
                   'best_val_loss': history['val_loss'][best_idx], 'best_val_acc': history['val_acc'][best_idx],
                   'best_val_macro_f1': history['val_macro_f1'][best_idx],
                   'best_val_weighted_penalty': history['val_weighted_penalty'][best_idx]},
                  fh, indent=2, ensure_ascii=False)
    print(f"\n[Fold {fold} KÉSZ] A finomhangolt 384-es BioCLIP elmentve ide: {best_weights_path}")

    del model, optimizer, train_loader, val_loader
    if DEVICE.type == "cuda": torch.cuda.empty_cache()

# ---------------------------------------------------------
# 7. FŐPROGRAM
# ---------------------------------------------------------
if __name__ == "__main__":
    print(f"Képek: {IMAGE_DIR} | CSV: {DATA_DIR} | Eszköz: {DEVICE}")
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, nargs="+")
    args, _ = parser.parse_known_args()

    folds = args.fold if args.fold else FOLDS_TO_RUN
    for fold in folds:
        if fold < 1 or fold > NUM_FOLDS:
            raise ValueError(f"Érvénytelen fold: {fold}. Engedélyezett tartomány: 1-{NUM_FOLDS}.")
    print(f"Tanítandó fold(ok): {folds}")

    for fold in folds:
        train_fold(fold)
    print(f"\n=== KÉSZ. Lefutott foldok: {folds} ===")
