import os
import sys
import json
import argparse
import shutil
import zipfile
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras import layers, models, Input, mixed_precision
from tensorflow.keras.optimizers import Adam

# Véletlenszám-generátorok fixálása a reprodukálhatóságért
tf.random.set_seed(42)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# ---------------------------------------------------------
# 0. KOMPATIBILITÁS ÉS HARDVERES GYORSÍTÁS (MIXED FP16)
# ---------------------------------------------------------
try:
    register_serializable = tf.keras.utils.register_keras_serializable
except AttributeError:
    from keras.saving import register_keras_serializable as register_serializable

mixed_precision.set_global_policy('mixed_float16')
print("Aktív számítási pontosság:", mixed_precision.global_policy().name)

# ---------------------------------------------------------
# 1. KÖRNYEZET ÉS HELYI SSD (/content) BEÁLLÍTÁSA
# ---------------------------------------------------------
IN_COLAB = False
try:
    from google.colab import drive
    IN_COLAB = True
except ImportError:
    pass

if IN_COLAB:
    drive.mount('/content/drive')
    DRIVE_DIR = Path('/content/drive/MyDrive/Mamba')
    
    ZIP_PATH = DRIVE_DIR / 'Databank/Train data/train_images_224_letterboxed.zip'
    EXTRACT_PATH = Path('/content/Databank/Train data')
    
    FOLDS_SOURCE_DIR = DRIVE_DIR / 'CSV files' / 'folds'
    LOGS_DIR = DRIVE_DIR / 'Logs'
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    if ZIP_PATH.exists():
        if not (EXTRACT_PATH / "train_images_224_letterboxed").exists():
            print(f"Képek kicsomagolása a helyi SSD-re: {EXTRACT_PATH} ...")
            EXTRACT_PATH.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
                zip_ref.extractall(EXTRACT_PATH)
            print("Kicsomagolás kész!")
    
    LOCAL_FOLDS_DIR = Path('/content/data_local/folds')
    LOCAL_FOLDS_DIR.mkdir(parents=True, exist_ok=True)
    
    if FOLDS_SOURCE_DIR.exists():
        for i in range(1, 6):
            for split in ['train', 'val']:
                file_name = f"fold_{i}_{split}.csv"
                src_csv = FOLDS_SOURCE_DIR / file_name
                dst_csv = LOCAL_FOLDS_DIR / file_name
                if src_csv.exists() and not dst_csv.exists():
                    shutil.copy2(src_csv, dst_csv)
    
    DATA_DIR = LOCAL_FOLDS_DIR if (LOCAL_FOLDS_DIR / "fold_1_train.csv").exists() else FOLDS_SOURCE_DIR
    IMAGE_DIR = EXTRACT_PATH / "train_images_224_letterboxed" if (EXTRACT_PATH / "train_images_224_letterboxed").exists() else EXTRACT_PATH
else:
    PROJECT_ROOT = Path().resolve().parents[1]
    DATA_DIR = PROJECT_ROOT / "data" / "CSV files" / "folds"
    IMAGE_DIR = PROJECT_ROOT / "data" / "Databank" / "Train data" / "train_images_224_letterboxed"
    LOGS_DIR = PROJECT_ROOT / "models"
    DRIVE_DIR = PROJECT_ROOT / "models"
    os.makedirs(LOGS_DIR, exist_ok=True)

# ---------------------------------------------------------
# HIPERPARAMÉTEREK ÉS FOLYAMAT-VEZÉRLÉS
# ---------------------------------------------------------
BATCH_SIZE = 128
WARMUP_EPOCHS = 10 
FINETUNE_EPOCHS = 30

WARMUP_LR = 5e-4      
FINETUNE_LR = 1.5e-4
NUM_FOLDS = 5
EARLY_STOPPING_PATIENCE = 3
EARLY_STOPPING_MIN_DELTA = 1e-3

# --- ITT ÁLLÍTSD BE, HOGY MELYIK FOLD(OK) FUSSON(AK)! ---
FOLDS_TO_RUN = [1]
# ---------------------------------------------------------

# TF Hub ResNet50 iNaturalist link (feature vector)
INAT_RESNET_URL = "https://tfhub.dev/google/inaturalist/resnet_v1_50/feature_vector/5"

# ---------------------------------------------------------
# 2. GYORS ADATBETÖLTŐ PIPELINE
# ---------------------------------------------------------
def process_image(file_path, class_id, venomous):
    img = tf.io.read_file(file_path)
    img = tf.io.decode_jpeg(img, channels=3, dct_method='INTEGER_FAST')
    img.set_shape([224, 224, 3])
    return img, {'species_head': class_id, 'venomous_head': venomous}

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal", dtype='float32'),
    layers.RandomRotation(0.15, fill_mode="reflect", dtype='float32'),
    layers.RandomZoom(0.1, fill_mode="reflect", dtype='float32'),
], name="gpu_augmentation")

def create_dataset(csv_file, img_dir, batch_size, is_training=True):
    df = pd.read_csv(csv_file)
    img_dir_str = str(img_dir)
    file_paths = [os.path.join(img_dir_str, str(p)) for p in df['image_path']]
    ds = tf.data.Dataset.from_tensor_slices((
        file_paths, df['class_id'].values.astype('int32'), df['MIVS'].values.astype('float32')
    ))
    options = tf.data.Options()
    options.experimental_deterministic = not is_training
    ds = ds.with_options(options)

    if is_training: ds = ds.shuffle(buffer_size=len(df), reshuffle_each_iteration=True)
    ds = ds.map(process_image, num_parallel_calls=tf.data.AUTOTUNE)
    if not is_training: ds = ds.cache()
    return ds.batch(batch_size, drop_remainder=is_training).prefetch(tf.data.AUTOTUNE)

# ---------------------------------------------------------
# 3. EGYEDI LOSS ÉS METRIKA FÜGGVÉNYEK
# ---------------------------------------------------------
v_mapping = tf.constant([
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
], dtype=tf.int32)
v_mapping_float = tf.cast(v_mapping, tf.float32)

@register_serializable()
def venomous_penalty_metric(y_true, y_pred):
    y_true_class = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
    y_pred_class = tf.cast(tf.argmax(y_pred, axis=-1), tf.int32)
    v_y = tf.gather(v_mapping_float, y_true_class)
    v_y_hat = tf.gather(v_mapping_float, y_pred_class)
    base_penalty = 1.0 + 4.0 * v_y + (1.0 - 4.0 * v_y) * v_y_hat
    return tf.reduce_mean(base_penalty * tf.cast(tf.not_equal(y_true_class, y_pred_class), tf.float32))

@register_serializable()
def cost_sensitive_species_loss(y_true, y_pred):
    y_pred = tf.cast(y_pred, tf.float32)
    y_true_int = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
    v_y = tf.gather(v_mapping_float, y_true_int)
    v_y_pred = tf.reduce_sum(y_pred * v_mapping_float, axis=-1)
    weight = v_y * (5.0 * (1.0 - v_y_pred) + 2.0 * v_y_pred) + (1.0 - v_y) * (1.0 * (1.0 - v_y_pred) + 2.0 * v_y_pred)
    scce = tf.keras.losses.sparse_categorical_crossentropy(y_true_int, y_pred)
    return tf.reduce_mean(scce * weight)

@register_serializable()
def weighted_binary_venomous_loss(y_true, y_pred):
    y_true = tf.cast(tf.reshape(y_true, [-1, 1]), tf.float32)
    y_pred = tf.cast(tf.reshape(y_pred, [-1, 1]), tf.float32)
    weights = tf.reshape(1.0 + 4.0 * y_true, [-1])
    return tf.reduce_mean(tf.keras.losses.binary_crossentropy(y_true, y_pred) * weights)

# ---------------------------------------------------------
# 4. MODELL ÉPÍTÉSE (TF HUB RESNET50)
# ---------------------------------------------------------
def build_model(num_species=296):
    inputs = Input(shape=(224, 224, 3), dtype=tf.uint8)
    
    # A ResNet v1 modellek [0, 1] közötti normalizált bemenetet várnak!
    x = layers.Rescaling(scale=1./255, dtype='float32')(inputs)
    x = data_augmentation(x)

    print(f"\n[+] Backbone letöltése/betöltése TF Hub-ról (ResNet50 iNaturalist)...")
    # TF Hub KerasLayer használata, kezdetben befagyasztva
    backbone = hub.KerasLayer(INAT_RESNET_URL, trainable=False, name='resnet50_inat')
    
    # A feature_vector modellek már tartalmazzák a GlobalAveragePooling-ot!
    x = backbone(x)
    
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.3)(x)

    s_out = layers.Dense(num_species, activation='softmax', dtype='float32', name='species_head')(x)
    v_out = layers.Dense(1, activation='sigmoid', dtype='float32', name='venomous_head')(x)
    return models.Model(inputs=inputs, outputs=[s_out, v_out]), backbone

def unfreeze_model_top(backbone):
    """
    TF Hub layer esetén a teljes gerinchálózatot trainable-re állítjuk,
    mivel a model-t nem tudjuk rétegenként feloldani. A finomhangolás kis LR-rel történik.
    """
    backbone.trainable = True

# ---------------------------------------------------------
# 5. SEGÉDFÜGGVÉNYEK (Grafikon és JSON)
# ---------------------------------------------------------
def plot_training_history(history, save_path, fold):
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Tanítási Loss', color='blue', linewidth=2)
    plt.plot(history.history['val_loss'], label='Validációs Loss', color='orange', linewidth=2)
    plt.title(f'Modell Költségfüggvény (Fold {fold})')
    plt.xlabel('Epoch'); plt.ylabel('Teljes Loss'); plt.legend(); plt.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(save_path)
    print(f"\n[+] Loss grafikon elmentve: {save_path}")

def save_fold_result(history, result_path, fold):
    key = 'val_loss'
    if history is None or key not in history.history: return
    best_idx = int(np.argmin(history.history[key]))
    result = {'fold': fold, 'best_epoch_in_this_run': best_idx + 1}
    for k, values in history.history.items():
        if k.startswith('val_'): result[k] = float(values[best_idx])
    with open(result_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

def create_early_stopping():
    return tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        mode='min',
        patience=EARLY_STOPPING_PATIENCE,
        min_delta=EARLY_STOPPING_MIN_DELTA,
        restore_best_weights=True,
        verbose=1
    )

def create_best_model_checkpoint(model_path):
    return tf.keras.callbacks.ModelCheckpoint(
        str(model_path),
        monitor='val_loss',
        mode='min',
        save_best_only=True,
        verbose=1
    )

# ---------------------------------------------------------
# 6. EGY FOLD BIZTONSÁGOS TANÍTÁSA
# ---------------------------------------------------------
def train_fold(fold):
    print("\n" + "="*60)
    print(f"=== {fold}. FOLD TANÍTÁSA INDUL (ResNet50 TF Hub) ===")
    print("="*60)

    train_csv = DATA_DIR / f"fold_{fold}_train.csv"
    val_csv = DATA_DIR / f"fold_{fold}_val.csv"
    
    # Fájlnevek frissítve resnet-re
    fold_model_path = DRIVE_DIR / f"snake_resnet50_model_fold_{fold}.keras"
    fold_warmup_model_path = DRIVE_DIR / f"snake_resnet50_warmup_fold_{fold}.keras"
    fold_warmup_log = LOGS_DIR / f"resnet_warmup_log_fold_{fold}.csv"
    fold_finetune_log = LOGS_DIR / f"resnet_finetune_log_fold_{fold}.csv"
    fold_plot_path = LOGS_DIR / f"resnet_training_loss_plot_fold_{fold}.png"
    fold_result_path = LOGS_DIR / f"resnet_fold_{fold}_result.json"

    train_ds = create_dataset(train_csv, IMAGE_DIR, BATCH_SIZE, is_training=True)
    val_ds = create_dataset(val_csv, IMAGE_DIR, BATCH_SIZE, is_training=False)

    compile_kwargs = {
        'loss': {'species_head': cost_sensitive_species_loss, 'venomous_head': weighted_binary_venomous_loss},
        'loss_weights': {'species_head': 1.0, 'venomous_head': 0.2},
        'metrics': {'species_head': ['accuracy', venomous_penalty_metric], 'venomous_head': 'accuracy'},
        'jit_compile': False
    }

    finetune_callbacks = [
        create_best_model_checkpoint(fold_model_path),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            mode='min',
            factor=0.3,
            patience=2,
            min_lr=1e-6,
            min_delta=EARLY_STOPPING_MIN_DELTA,
            verbose=1
        ),
        create_early_stopping(),
        tf.keras.callbacks.CSVLogger(str(fold_finetune_log), append=True)
    ]

    # Szótár a Custom Objects betöltéséhez a saved model-nél
    custom_objs = {
        'venomous_penalty_metric': venomous_penalty_metric,
        'cost_sensitive_species_loss': cost_sensitive_species_loss,
        'weighted_binary_venomous_loss': weighted_binary_venomous_loss,
        'KerasLayer': hub.KerasLayer
    }

    history = None

    if fold_model_path.exists():
        print(f"\n[+] Meglévő modell betöltése: {fold_model_path}")
        model = tf.keras.models.load_model(str(fold_model_path), custom_objects=custom_objs)
        backbone = model.get_layer('resnet50_inat')
        print(f"[!] 1. FÁZIS (Warm-up) átugorva. 2. FÁZIS FOLYTATÁSA ({FINETUNE_EPOCHS} epoch)...")
        unfreeze_model_top(backbone)
        model.compile(optimizer=Adam(learning_rate=FINETUNE_LR), **compile_kwargs)
        history = model.fit(train_ds, validation_data=val_ds, epochs=FINETUNE_EPOCHS, callbacks=finetune_callbacks)
    else:
        if fold_warmup_model_path.exists():
            print(f"\n[+] Kész Warm-up modell betöltése: {fold_warmup_model_path}")
            model = tf.keras.models.load_model(str(fold_warmup_model_path), custom_objects=custom_objs)
            backbone = model.get_layer('resnet50_inat')
        else:
            model, backbone = build_model()
            print(f"\n=== 1. FÁZIS: Bemelegítés (Fold {fold}, {WARMUP_EPOCHS} epoch) ===")
            model.compile(optimizer=tf.keras.optimizers.AdamW(learning_rate=WARMUP_LR, weight_decay=1e-4, clipnorm=1.0), **compile_kwargs)
            warmup_callbacks = [
                create_best_model_checkpoint(fold_warmup_model_path),
                create_early_stopping(),
                tf.keras.callbacks.CSVLogger(str(fold_warmup_log), append=True),
            ]
            model.fit(train_ds, validation_data=val_ds, epochs=WARMUP_EPOCHS, callbacks=warmup_callbacks)

        print(f"\n=== 2. FÁZIS: Finomhangolás (Fold {fold}, {FINETUNE_EPOCHS} epoch) ===")
        unfreeze_model_top(backbone)
        model.compile(optimizer=Adam(learning_rate=FINETUNE_LR), **compile_kwargs)
        history = model.fit(train_ds, validation_data=val_ds, epochs=WARMUP_EPOCHS + FINETUNE_EPOCHS, initial_epoch=WARMUP_EPOCHS, callbacks=finetune_callbacks)

    print(f"\n[Fold {fold} KÉSZ] A legjobb modell elmentve ide: {fold_model_path}")
    if history is not None:
        plot_training_history(history, str(fold_plot_path), fold)
        save_fold_result(history, str(fold_result_path), fold)

    del model, backbone, train_ds, val_ds
    tf.keras.backend.clear_session()

# ---------------------------------------------------------
# 7. FŐ CIKLUS INDÍTÁSA
# ---------------------------------------------------------
if __name__ == "__main__":
    print(f"Képek beolvasása innen: {IMAGE_DIR}")
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, nargs="+")
    args, _ = parser.parse_known_args()

    folds = args.fold if args.fold else FOLDS_TO_RUN
    
    print(f"Tanítandó fold(ok): {folds}")
    for f in folds:
        train_fold(f)
    print(f"\n=== KÉSZ. Lefutott foldok: {folds} ===")