import os
import json
import shutil
import zipfile
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import open_clip
from sklearn.metrics import f1_score

# ---------------------------------------------------------
# 1. KÖRNYEZET ÉS ADATÚTVONALAK
# ---------------------------------------------------------
IN_COLAB = True
try:
    from google.colab import drive
    drive.mount('/content/drive')
except:
    IN_COLAB = False

DRIVE_DIR = Path('/content/drive/MyDrive/Mamba') if IN_COLAB else Path('./')
ZIP_PATH = DRIVE_DIR / 'Databank/Train data/train_images_224_letterboxed.zip'
EXTRACT_PATH = Path('/content/Databank/Train data')
FOLDS_SOURCE_DIR = DRIVE_DIR / 'CSV files' / 'folds'
LOGS_DIR = DRIVE_DIR / 'Logs'
LOGS_DIR.mkdir(parents=True, exist_ok=True)

if ZIP_PATH.exists() and not (EXTRACT_PATH / "train_images_224_letterboxed").exists():
    print("[+] Képek kicsomagolása a helyi SSD-re...")
    EXTRACT_PATH.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
        zip_ref.extractall(EXTRACT_PATH)
    print("Kicsomagolás kész!")

LOCAL_FOLDS_DIR = Path('/content/data_local/folds')
LOCAL_FOLDS_DIR.mkdir(parents=True, exist_ok=True)
if FOLDS_SOURCE_DIR.exists():
    for i in range(1, 6):
        for split in ["train", "val"]:
            f = f"fold_{i}_{split}.csv"
            src, dst = FOLDS_SOURCE_DIR / f, LOCAL_FOLDS_DIR / f
            if src.exists() and not dst.exists(): shutil.copy2(src, dst)

DATA_DIR = LOCAL_FOLDS_DIR if (LOCAL_FOLDS_DIR / "fold_1_train.csv").exists() else FOLDS_SOURCE_DIR
IMAGE_DIR = EXTRACT_PATH / "train_images_224_letterboxed"

# ---------------------------------------------------------
# 2. HIPERPARAMÉTEREK
# ---------------------------------------------------------
NUM_CLASSES = 296
BATCH_SIZE = 128

WARMUP_EPOCHS = 10
FINETUNE_EPOCHS = 30
WARMUP_LR = 7.27e-03
FINETUNE_LR = 1.41e-06

FOLDS_TO_RUN = [1]
EARLY_STOPPING_PATIENCE = 5
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
V_MAPPING_NP = np.array(v_mapping_list)

# ---------------------------------------------------------
# 3. DATASET ÉS AUGMENTÁCIÓ
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

# BioCLIP (OpenCLIP) normalizációs értékek
normalize = transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(), transforms.RandomRotation(15), transforms.ToTensor(), normalize
])
val_transform = transforms.Compose([transforms.ToTensor(), normalize])

# ---------------------------------------------------------
# 4. MODELL ÉS KÖLTSÉGFÜGGVÉNYEK
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
# 5. VALIDÁCIÓS SEGÉDFÜGGVÉNY
# ---------------------------------------------------------
def evaluate_model(model, val_loader):
    model.eval()
    val_loss = 0.0
    all_preds_s, all_targets_s = [], []
    all_preds_v, all_targets_v = [], []

    with torch.no_grad():
        for images, species, venom in val_loader:
            images, species, venom = images.to(DEVICE), species.to(DEVICE), venom.to(DEVICE)
            s_logits, v_logits = model(images)
            loss = compute_loss(s_logits, v_logits, species, venom)
            val_loss += loss.item()
            
            all_preds_s.append(s_logits.argmax(dim=-1).cpu().numpy())
            all_targets_s.append(species.cpu().numpy())
            all_preds_v.append((torch.sigmoid(v_logits) > 0.5).int().cpu().numpy().flatten())
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
    plt.title(f"BioCLIP Költségfüggvény (Fold {fold} - Finomhangolás)")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend(); plt.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(save_path, dpi=150); plt.close()

# ---------------------------------------------------------
# 6. KÉTFÁZISÚ TANÍTÁSI FOLYAMAT FOLDONKÉNT
# ---------------------------------------------------------
def train_fold(fold):
    print("\n" + "=" * 65)
    print(f"=== {fold}. FOLD: BioCLIP Kétfázisú Tanítás (224x224) ===")
    print("=" * 65)

    BIOCLIP_MODELS_DIR = DRIVE_DIR / 'models' / 'bioclip'
    BIOCLIP_MODELS_DIR.mkdir(parents=True, exist_ok=True)

    warmup_weights_path = BIOCLIP_MODELS_DIR / f"bioclip_warmup_fold_{fold}.pth"
    finetune_weights_path = BIOCLIP_MODELS_DIR / f"bioclip_model_fold_{fold}.pth"
    
    warmup_log_path = LOGS_DIR / f"bioclip_warmup_log_fold_{fold}.csv"
    finetune_log_path = LOGS_DIR / f"bioclip_finetune_log_fold_{fold}.csv"
    plot_path = LOGS_DIR / f"bioclip_plot_fold_{fold}.png"

    train_ds = SnakeDataset(DATA_DIR / f"fold_{fold}_train.csv", IMAGE_DIR, transform=train_transform)
    val_ds = SnakeDataset(DATA_DIR / f"fold_{fold}_val.csv", IMAGE_DIR, transform=val_transform)
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

    model = BioCLIPMultiTask(NUM_CLASSES).to(DEVICE)
    csv_header = "epoch,train_loss,val_loss,val_acc,val_macro_f1,val_venom_acc,val_weighted_penalty\n"

    # --- 1. FÁZIS: BEMELEGÍTÉS (WARM-UP) ---
    if finetune_weights_path.exists():
        print(f"\n[+] Kész végső modell megtalálva: {finetune_weights_path}")
        return
    elif warmup_weights_path.exists():
        print(f"\n[+] Meglévő Warm-up súlyok betöltése: {warmup_weights_path}")
        model.load_state_dict(torch.load(warmup_weights_path, map_location=DEVICE))
    else:
        print(f"\n=== 1. FÁZIS: Bemelegítés (Fagyasztott Backbone, {WARMUP_EPOCHS} epoch) ===")
        # Backbone fagyasztása
        for param in model.backbone.parameters():
            param.requires_grad = False
            
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=WARMUP_LR, weight_decay=1e-4)
        
        with open(warmup_log_path, 'w') as f: f.write(csv_header)
        best_warmup_loss = float('inf')

        for epoch in range(WARMUP_EPOCHS):
            model.train()
            train_loss = 0.0
            for images, species, venom in train_loader:
                images, species, venom = images.to(DEVICE), species.to(DEVICE), venom.to(DEVICE)
                optimizer.zero_grad()
                s_logits, v_logits = model(images)
                loss = compute_loss(s_logits, v_logits, species, venom)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            t_loss_avg = train_loss / len(train_loader)
            v_loss_avg, v_acc, v_f1, v_venom, v_pen = evaluate_model(model, val_loader)

            with open(warmup_log_path, 'a') as f: 
                f.write(f"{epoch+1},{t_loss_avg:.6f},{v_loss_avg:.6f},{v_acc:.6f},{v_f1:.6f},{v_venom:.6f},{v_pen:.6f}\n")

            print(f"Warm-up {epoch+1:02d} | Train Loss: {t_loss_avg:.4f} | Val Loss: {v_loss_avg:.4f} | Acc: {v_acc:.4f}")
            
            if v_loss_avg < best_warmup_loss:
                best_warmup_loss = v_loss_avg
                torch.save(model.state_dict(), warmup_weights_path)

        # Betöltjük a legjobb warmup állapotot a finomhangolás előtt
        model.load_state_dict(torch.load(warmup_weights_path, map_location=DEVICE))

    # --- 2. FÁZIS: FINOMHANGOLÁS (FINE-TUNING) ---
    print(f"\n=== 2. FÁZIS: Teljes Finomhangolás (Kiolvasztott Backbone, {FINETUNE_EPOCHS} epoch) ===")
    
    # Backbone kiolvasztása
    for param in model.backbone.parameters():
        param.requires_grad = True

    optimizer = optim.AdamW(model.parameters(), lr=FINETUNE_LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.3, patience=2, min_lr=1e-6)
    
    with open(finetune_log_path, 'w') as f: f.write(csv_header)
    
    best_val_loss = float('inf')
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': []}

    for epoch in range(FINETUNE_EPOCHS):
        model.train()
        train_loss = 0.0
        for images, species, venom in train_loader:
            images, species, venom = images.to(DEVICE), species.to(DEVICE), venom.to(DEVICE)
            optimizer.zero_grad()
            s_logits, v_logits = model(images)
            loss = compute_loss(s_logits, v_logits, species, venom)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        t_loss_avg = train_loss / len(train_loader)
        v_loss_avg, v_acc, v_f1, v_venom, v_pen = evaluate_model(model, val_loader)

        history['train_loss'].append(t_loss_avg)
        history['val_loss'].append(v_loss_avg)

        with open(finetune_log_path, 'a') as f: 
            f.write(f"{epoch+1},{t_loss_avg:.6f},{v_loss_avg:.6f},{v_acc:.6f},{v_f1:.6f},{v_venom:.6f},{v_pen:.6f}\n")

        print(f"Fine-tune {epoch+1:02d} | Train Loss: {t_loss_avg:.4f} | Val Loss: {v_loss_avg:.4f} | Acc: {v_acc:.4f} | F1: {v_f1:.4f} | Pen: {v_pen:.4f}")
        
        scheduler.step(v_loss_avg)
        
        if v_loss_avg < best_val_loss:
            best_val_loss = v_loss_avg
            patience_counter = 0
            torch.save(model.state_dict(), finetune_weights_path)
            print("  [+] Best model mentve!")
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print(f"  [!] Early Stopping aktiválódott a(z) {epoch+1}. epochnál!")
                break

    plot_history(history, plot_path, fold)
    print(f"\n[Fold {fold} KÉSZ] A finomhangolt 224-es BioCLIP elmentve ide: {finetune_weights_path}")
    
    del model, train_loader, val_loader
    torch.cuda.empty_cache()

# ---------------------------------------------------------
# 7. FŐPROGRAM
# ---------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, nargs="+")
    args, _ = parser.parse_known_args()

    folds = args.fold if args.fold else FOLDS_TO_RUN
    for fold in folds:
        train_fold(fold)