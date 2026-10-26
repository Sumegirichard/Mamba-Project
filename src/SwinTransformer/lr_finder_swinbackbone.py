# -*- coding: utf-8 -*-
"""
LR finder a FINOMHANGOLÁSI fázishoz (Swin Transformer, kiolvasztott backbone).
Betölti a kész Warm-up súlyokat, majd alacsonyabb sávban (1e-7 -> 1e-2) keresi az optimális FINETUNE_LR-t.
A kiírt FINETUNE_LR értéket írd be a model_swin_224.py FINETUNE_LR változójába.
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import sys
import json
import shutil
import zipfile
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, Input, mixed_precision
from transformers import TFSwinModel

SEED = 42
tf.keras.utils.set_random_seed(SEED)

# ---------------------------------------------------------
# 0. KOMPATIBILITÁS ÉS HARDVERES GYORSÍTÁS
# ---------------------------------------------------------
try:
    register_serializable = tf.keras.utils.register_keras_serializable
except AttributeError:
    from keras.saving import register_keras_serializable as register_serializable

mixed_precision.set_global_policy('mixed_float16')
print("Aktív számítási pontosság:", mixed_precision.global_policy().name)

# ---------------------------------------------------------
# 1. KÖRNYEZET BEÁLLÍTÁSA
# ---------------------------------------------------------
IMAGE_FOLDER_NAME = "train_images_224_letterboxed"
CSV_NAME = "fold_1_train.csv"

IN_COLAB = False
try:
    from google.colab import drive
    IN_COLAB = True
except ImportError:
    pass

if IN_COLAB:
    drive.mount('/content/drive')
    DRIVE_DIR = Path('/content/drive/MyDrive/Mamba')
    ZIP_PATH = DRIVE_DIR / f'Databank/Train data/{IMAGE_FOLDER_NAME}.zip'
    EXTRACT_PATH = Path('/content/Databank/Train data')
    FOLDS_SOURCE_DIR = DRIVE_DIR / 'CSV files' / 'folds'
    LOGS_DIR = DRIVE_DIR / 'Logs'
    SWIN_MODELS_DIR = DRIVE_DIR / 'models' / 'swin'
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    if ZIP_PATH.exists() and not (EXTRACT_PATH / IMAGE_FOLDER_NAME).exists():
        EXTRACT_PATH.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
            zip_ref.extractall(EXTRACT_PATH)

    LOCAL_FOLDS_DIR = Path('/content/data_local/folds')
    LOCAL_FOLDS_DIR.mkdir(parents=True, exist_ok=True)
    src_csv = FOLDS_SOURCE_DIR / CSV_NAME
    dst_csv = LOCAL_FOLDS_DIR / CSV_NAME
    if src_csv.exists() and not dst_csv.exists():
        shutil.copy2(src_csv, dst_csv)

    DATA_DIR = LOCAL_FOLDS_DIR if dst_csv.exists() else FOLDS_SOURCE_DIR
    IMAGE_DIR = EXTRACT_PATH / IMAGE_FOLDER_NAME if (EXTRACT_PATH / IMAGE_FOLDER_NAME).exists() else EXTRACT_PATH
else:
    PROJECT_ROOT = Path().resolve().parents[1]
    DATA_DIR = PROJECT_ROOT / "data" / "CSV files" / "folds"
    IMAGE_DIR = PROJECT_ROOT / "data" / "Databank" / "Train data" / IMAGE_FOLDER_NAME
    LOGS_DIR = PROJECT_ROOT / "models"
    SWIN_MODELS_DIR = PROJECT_ROOT / "models" / "swin"
    os.makedirs(LOGS_DIR, exist_ok=True)

WARMUP_WEIGHTS_PATH = SWIN_MODELS_DIR / "snake_swin_warmup_fold_1.weights.h5"
PLOT_PATH = LOGS_DIR / "lr_finder_swin_finetune_plot.png"
JSON_PATH = LOGS_DIR / "lr_finder_swin_finetune.json"

# ---------------------------------------------------------
# 2. FINOMHANGOLÁSI BEÁLLÍTÁSOK (Alacsonyabb keresési sáv)
# ---------------------------------------------------------
NUM_CLASSES = 296
BATCH_SIZE = 128          
SWEEP_STEPS = 200         
N_RUNS = 2                
START_LR = 1e-7           # Finomhangoláshoz a tartomány lejjebb kerül
END_LR = 1e-2
LR_CLIP = (1e-6, 1e-3)        
DEFAULT_LR = 1e-4  
SMOOTH_BETA = 0.85        
DIVERGENCE_FACTOR = 4.0   
MIN_DIV = 3.0             

# ---------------------------------------------------------
# 3. GYORS ADATBETÖLTŐ PIPELINE
# ---------------------------------------------------------
def process_image(file_path, class_id, venomous):
    img = tf.io.read_file(file_path)
    img = tf.io.decode_jpeg(img, channels=3, dct_method='INTEGER_FAST')
    img.set_shape([224, 224, 3])
    return img, {'species_head': class_id, 'venomous_head': venomous}

def resolve_paths(image_paths, img_dir):
    img_dir = str(img_dir)
    full = [os.path.join(img_dir, str(p)) for p in image_paths]
    if full and not os.path.exists(full[0]):
        alt = [os.path.join(img_dir, os.path.basename(str(p))) for p in image_paths]
        if os.path.exists(alt[0]): return alt
        raise FileNotFoundError(f"Az első kép sem található: {full[0]}")
    return full

def create_train_dataset(df, img_dir, batch_size, seed):
    file_paths = resolve_paths(df['image_path'].tolist(), img_dir)
    ds = tf.data.Dataset.from_tensor_slices((file_paths, df['class_id'].values.astype('int32'), df['MIVS'].values.astype('float32')))
    options = tf.data.Options()
    if hasattr(options, 'deterministic'): options.deterministic = False
    else: options.experimental_deterministic = False
    ds = ds.with_options(options)
    ds = ds.shuffle(buffer_size=len(df), seed=seed, reshuffle_each_iteration=True)
    ds = ds.map(process_image, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size, drop_remainder=True).repeat().prefetch(tf.data.AUTOTUNE)
    return ds

# ---------------------------------------------------------
# 4. LOSS FÜGGVÉNYEK
# ---------------------------------------------------------
V_MAPPING_NP = None
v_mapping_float = None

def init_venom_mapping(df):
    global V_MAPPING_NP, v_mapping_float
    v = np.zeros(NUM_CLASSES, dtype=np.int32)
    per_class = df.groupby('class_id')['MIVS'].first().astype(int)
    v[per_class.index.values] = per_class.values
    V_MAPPING_NP = v
    v_mapping_float = tf.constant(v, dtype=tf.float32)

@register_serializable()
def cost_sensitive_species_loss(y_true, y_pred):
    y_pred = tf.cast(y_pred, tf.float32)
    y_true_int = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
    v_y = tf.gather(v_mapping_float, y_true_int)
    v_y_pred = tf.reduce_sum(y_pred * v_mapping_float, axis=-1)
    weight_if_venomous = 5.0 * (1.0 - v_y_pred) + 2.0 * v_y_pred
    weight_if_non_venomous = 1.0 * (1.0 - v_y_pred) + 2.0 * v_y_pred
    weight = v_y * weight_if_venomous + (1.0 - v_y) * weight_if_non_venomous
    scce = tf.keras.losses.sparse_categorical_crossentropy(y_true_int, y_pred)
    return tf.reduce_mean(scce * weight)

@register_serializable()
def weighted_binary_venomous_loss(y_true, y_pred):
    y_true = tf.cast(tf.reshape(y_true, [-1, 1]), tf.float32)
    y_pred = tf.cast(tf.reshape(y_pred, [-1, 1]), tf.float32)
    weights = tf.reshape(1.0 + 4.0 * y_true, [-1])
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    return tf.reduce_mean(bce * weights)

COMPILE_KWARGS = {
    'loss': {'species_head': cost_sensitive_species_loss, 'venomous_head': weighted_binary_venomous_loss},
    'loss_weights': {'species_head': 1.0, 'venomous_head': 0.2},
    'metrics': {'species_head': ['accuracy'], 'venomous_head': ['accuracy']},
    'jit_compile': False
}

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
    
    out.update(ok=True, lr_steepest=lr_steepest, lr_min_loss=lr_min_loss, recommended_lr=rec, min_smoothed_loss=float(sm[idx_min]))
    return out

def aggregate_runs(analyses, default_lr, clip=LR_CLIP):
    good = [a for a in analyses if a.get('ok')]
    if not good: return float(default_lr), ["Nincs értékelhető futás."]
    recs = np.array([a['recommended_lr'] for a in good], dtype=np.float64)
    final = float(np.clip(float(np.exp(np.mean(np.log(recs)))), clip[0], clip[1]))
    return final, []

class LRFinder(tf.keras.callbacks.Callback):
    def __init__(self, start_lr, end_lr, max_steps, beta=SMOOTH_BETA, divergence_factor=DIVERGENCE_FACTOR):
        super().__init__()
        self.start_lr = start_lr; self.end_lr = end_lr; self.max_steps = max_steps
        self.beta = beta; self.divergence_factor = divergence_factor
        self.factor = (end_lr / start_lr) ** (1.0 / max_steps)
        self.lrs = []; self.losses = []; self.smoothed = []
        self.best_smooth = float('inf'); self.step = 0
        self.diverged_at = None; self._prev_running = 0.0

    def _set_lr(self, lr):
        opt = self.model.optimizer
        target = getattr(opt, 'inner_optimizer', opt)
        if hasattr(target.learning_rate, 'assign'): target.learning_rate.assign(lr)
        else: target.learning_rate = lr

    def on_train_batch_begin(self, batch, logs=None):
        lr = self.start_lr * (self.factor ** self.step)
        self._set_lr(lr)
        self.lrs.append(lr)

    def on_train_batch_end(self, batch, logs=None):
        if not logs or logs.get('loss') is None: return
        running = float(logs['loss'])
        n = self.step
        loss = running if n == 0 else (n + 1) * running - n * self._prev_running
        self._prev_running = running
        self.losses.append(loss)
        self.step += 1

        if not np.isfinite(loss):
            self.diverged_at = self.step
            self.model.stop_training = True
            return

        smooth = loss if n == 0 else self.beta * self.smoothed[-1] + (1.0 - self.beta) * loss
        self.smoothed.append(smooth)
        self.best_smooth = min(self.best_smooth, smooth)

        if self.step > 20 and smooth > self.divergence_factor * self.best_smooth:
            self.diverged_at = self.step
            self.model.stop_training = True

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
    plt.savefig(save_path, dpi=130); plt.close()

def report_and_save(analyses, final_lr, notes, json_key, constant_name):
    print("\n" + "=" * 60)
    print(f"--> VÉGSŐ ÉRTÉK: {final_lr:.2e}")
    print(f"--> Ezt ({final_lr:.6f}) írd be a fő tanító kód {constant_name} változójába!")
    print("=" * 60)
    with open(JSON_PATH, 'w', encoding='utf-8') as fh: 
        json.dump({json_key: final_lr, 'runs': analyses, 'notes': notes}, fh, indent=2, ensure_ascii=False)

# ---------------------------------------------------------
# 6. SWIN MODELL ÉPÍTÉSE (FINOMHANGOLÁSHOZ KIOLVASZTVA)
# ---------------------------------------------------------
def make_augmentation():
    return tf.keras.Sequential([
        layers.RandomFlip("horizontal", dtype='float32'),
        layers.RandomRotation(0.15, fill_mode="reflect", dtype='float32'),
        layers.RandomZoom(0.1, fill_mode="reflect", dtype='float32'),
    ], name="gpu_augmentation")

def build_model_finetune(num_species=NUM_CLASSES):
    inputs = Input(shape=(224, 224, 3), dtype=tf.uint8)
    
    x = make_augmentation()(tf.cast(inputs, tf.float32))
    x = x / 255.0
    mean = tf.constant([0.485, 0.456, 0.406], shape=[1, 1, 1, 3], dtype=tf.float32)
    std = tf.constant([0.229, 0.224, 0.225], shape=[1, 1, 1, 3], dtype=tf.float32)
    x = (x - mean) / std

    backbone = TFSwinModel.from_pretrained("microsoft/swin-base-patch4-window7-224", from_pt=True)
    
    # FONTOS: A Transformer hálózat teljes kiolvasztása
    backbone.trainable = True

    outputs = backbone(pixel_values=x, training=False)
    x = outputs.pooler_output

    x = layers.Dense(512, activation='gelu')(x)
    x = layers.Dropout(0.3)(x)

    species_out = layers.Dense(num_species, activation='softmax', dtype='float32', name='species_head')(x)
    venomous_out = layers.Dense(1, activation='sigmoid', dtype='float32', name='venomous_head')(x)
    return models.Model(inputs=inputs, outputs=[species_out, venomous_out])

# ---------------------------------------------------------
# 7. FŐPROGRAM
# ---------------------------------------------------------
def main():
    if not WARMUP_WEIGHTS_PATH.exists():
        sys.exit(f"\n[!] HIBA: Nem található a bemelegített Swin súlyfájl itt: {WARMUP_WEIGHTS_PATH}\n"
                 f"Először futtasd le a model_swin_224.py 1. fázisát (Warm-up)!")

    train_df = pd.read_csv(DATA_DIR / CSV_NAME)
    init_venom_mapping(train_df)

    finders, analyses = [], []
    for run in range(N_RUNS):
        seed = SEED + run
        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(seed)

        train_ds = create_train_dataset(train_df, IMAGE_DIR, BATCH_SIZE, seed)
        
        # 1. Építés kiolvasztott backbone-nal
        model = build_model_finetune()
        
        # 2. Bemelegített súlyok visszatöltése (.weights.h5 formátum)
        print(f"\n[+] Warm-up súlyok betöltése a keresés előtti finomhangoláshoz: {WARMUP_WEIGHTS_PATH}")
        model.load_weights(str(WARMUP_WEIGHTS_PATH))

        model.compile(optimizer=tf.keras.optimizers.AdamW(learning_rate=START_LR, weight_decay=1e-4, clipnorm=1.0),
                      **COMPILE_KWARGS)

        finder = LRFinder(START_LR, END_LR, SWEEP_STEPS)
        print(f"=== {run + 1}/{N_RUNS}. futás (Finetune LR keresés) | seed={seed} ===")
        model.fit(train_ds, epochs=1, steps_per_epoch=SWEEP_STEPS, callbacks=[finder], verbose=1)

        analysis = analyze_sweep(finder.lrs, finder.losses, diverged=finder.diverged_at is not None)
        finders.append(finder); analyses.append(analysis)

    final_lr, notes = aggregate_runs(analyses, DEFAULT_LR)
    report_and_save(analyses, final_lr, notes, 'finetune_lr', 'FINETUNE_LR')
    plot_and_save(finders, analyses, final_lr, PLOT_PATH, 'LR Finder eredménye (Swin Transformer - Finetune)', 'Ajánlott Finetune LR')

if __name__ == "__main__":
    main()