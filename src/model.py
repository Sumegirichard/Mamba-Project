import os
import shutil
import zipfile
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, Input, mixed_precision
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.optimizers import Adam
# from random_seed import set_global_seed

# set_global_seed(42)
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
    
    # Új útvonal a foldokhoz a kép alapján
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
    else:
        print(f"FIGYELEM: A ZIP fájl nem található a Drive-on: {ZIP_PATH}")

    # Fold CSV-k másolása a helyi meghajtóra a sebességért
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
    
    if (EXTRACT_PATH / "train_images_224_letterboxed").exists():
        IMAGE_DIR = EXTRACT_PATH / "train_images_224_letterboxed"
    else:
        IMAGE_DIR = EXTRACT_PATH
else:
    try:
        PROJECT_ROOT = Path(__file__).resolve().parents[1]
    except NameError:
        PROJECT_ROOT = Path().resolve().parents[1]
    DATA_DIR = PROJECT_ROOT / "data" / "CSV files" / "folds"
    IMAGE_DIR = PROJECT_ROOT / "data" / "Databank" / "Train data" / "train_images_224_letterboxed"
    LOGS_DIR = PROJECT_ROOT / "models"
    os.makedirs(LOGS_DIR, exist_ok=True)
    DRIVE_DIR = PROJECT_ROOT / "models"

# Hiperparaméterek
BATCH_SIZE = 128
WARMUP_EPOCHS = 10 
FINETUNE_EPOCHS = 30
WARMUP_LR = 5e-4
FINETUNE_LR = 1.5e-4
NUM_FOLDS = 5

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
        file_paths,
        df['class_id'].values.astype('int32'),
        df['MIVS'].values.astype('float32')
    ))

    options = tf.data.Options()
    options.experimental_deterministic = not is_training
    ds = ds.with_options(options)

    if is_training:
        ds = ds.shuffle(buffer_size=len(df), reshuffle_each_iteration=True)

    ds = ds.map(process_image, num_parallel_calls=tf.data.AUTOTUNE)

    if not is_training:
        ds = ds.cache()

    ds = ds.batch(batch_size, drop_remainder=is_training)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds

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
    is_wrong = tf.cast(tf.not_equal(y_true_class, y_pred_class), tf.float32)
    return tf.reduce_mean(base_penalty * is_wrong)

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

@register_serializable()
def resnet_preprocess_fn(x):
    return preprocess_input(tf.cast(x, tf.float32))

# ---------------------------------------------------------
# 4. TRANSFER LEARNING MODELL ÉS GRAFIKON
# ---------------------------------------------------------
def build_model(num_species=296):
    inputs = Input(shape=(224, 224, 3), dtype=tf.uint8, name="uint8_input")
    x = layers.Rescaling(scale=1.0, dtype='float32', name="to_float32")(inputs)
    x = data_augmentation(x)
    x = layers.Lambda(resnet_preprocess_fn, output_shape=(224, 224, 3), name="resnet_preprocess")(x)

    backbone = ResNet50(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
    backbone.trainable = False

    x = backbone(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.3)(x)

    species_out = layers.Dense(num_species, activation='softmax', dtype='float32', name='species_head')(x)
    venomous_out = layers.Dense(1, activation='sigmoid', dtype='float32', name='venomous_head')(x)

    return models.Model(inputs=inputs, outputs=[species_out, venomous_out]), backbone

def unfreeze_resnet_top(backbone, num_layers=30):
    backbone.trainable = True
    for layer in backbone.layers[:-num_layers]:
        layer.trainable = False
    for layer in backbone.layers[-num_layers:]:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False
        else:
            layer.trainable = True

def plot_training_history(history, save_path, fold):
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Tanítási Loss', color='blue', linewidth=2)
    plt.plot(history.history['val_loss'], label='Validációs Loss', color='orange', linewidth=2)
    plt.title(f'Modell Költségfüggvény (Fold {fold})', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Teljes Loss', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"\n[+] A Loss grafikon sikeresen elmentve ide: {save_path}")

# ---------------------------------------------------------
# 5. FŐPROGRAM (K-Fold Keresztvalidáció)
# ---------------------------------------------------------
if __name__ == "__main__":
    print(f"Képek beolvasása innen: {IMAGE_DIR}")
    
    compile_kwargs = {
        'loss': {'species_head': cost_sensitive_species_loss, 'venomous_head': weighted_binary_venomous_loss},
        'loss_weights': {'species_head': 1.0, 'venomous_head': 0.2},
        'metrics': {'species_head': ['accuracy', venomous_penalty_metric], 'venomous_head': 'accuracy'},
        'jit_compile': False
    }

    for fold in range(1, NUM_FOLDS + 1):
        print("\n" + "="*60)
        print(f"=== {fold}. FOLD TANÍTÁSA INDUL ===")
        print("="*60)

        # Fájl útvonalak az aktuális foldhoz a kép alapján
        train_csv = DATA_DIR / f"fold_{fold}_train.csv"
        val_csv = DATA_DIR / f"fold_{fold}_val.csv"
        
        # Cél útvonalak logoláshoz és mentéshez
        fold_model_path = DRIVE_DIR / f"snake_multitask_model_fold_{fold}.keras"
        fold_warmup_model_path = DRIVE_DIR / f"snake_multitask_warmup_fold_{fold}.keras"
        fold_warmup_log = LOGS_DIR / f"warmup_log_fold_{fold}.csv"
        fold_finetune_log = LOGS_DIR / f"finetune_log_fold_{fold}.csv"
        fold_plot_path = LOGS_DIR / f"training_loss_plot_fold_{fold}.png"

        print(f"Train CSV: {train_csv}")
        print(f"Val CSV: {val_csv}")

        train_ds = create_dataset(train_csv, IMAGE_DIR, BATCH_SIZE, is_training=True)
        val_ds = create_dataset(val_csv, IMAGE_DIR, BATCH_SIZE, is_training=False)

        finetune_callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                filepath=str(fold_model_path),
                monitor='val_species_head_venomous_penalty_metric',
                mode='min',
                save_best_only=True,
                verbose=1
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_species_head_venomous_penalty_metric',
                mode='min',
                factor=0.3,
                patience=2,
                min_lr=1e-6,
                min_delta=1e-3,
                verbose=1
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor='val_species_head_venomous_penalty_metric',
                mode='min',
                patience=5,
                restore_best_weights=True,
                min_delta=1e-3,
                verbose=1
            ),
            tf.keras.callbacks.CSVLogger(filename=str(fold_finetune_log), separator=',', append=True)
        ]

        history = None

        if fold_model_path.exists():
            print(f"\n[+] Meglévő modell betöltése a {fold}. foldhoz: {fold_model_path}")
            model = tf.keras.models.load_model(str(fold_model_path))
            backbone = model.get_layer('resnet50')

            print(f"\n[!] Az 1. FÁZIST (Warm-up) átugorjuk.")
            print(f"2. FÁZIS: Finomhangolás FOLYTATÁSA ({FINETUNE_EPOCHS} epoch, Batch={BATCH_SIZE})...")

            unfreeze_resnet_top(backbone, num_layers=30)
            model.compile(optimizer=Adam(learning_rate=FINETUNE_LR), **compile_kwargs)
            history = model.fit(train_ds, validation_data=val_ds, epochs=FINETUNE_EPOCHS, callbacks=finetune_callbacks)

        else:
            if fold_warmup_model_path.exists():
                print(f"\n[+] Bemelegített (Warm-up) modell betöltése: {fold_warmup_model_path}")
                model = tf.keras.models.load_model(str(fold_warmup_model_path))
                backbone = model.get_layer('resnet50')
                print(f"\n[!] Az 1. FÁZIST (Warm-up) átugorjuk, mert már elkészült.")
                
            else:
                model, backbone = build_model()
                print(f"\n=== 1. FÁZIS: Új fej bemelegítése (Fold {fold}, {WARMUP_EPOCHS} epoch) ===")
                warmup_optimizer = tf.keras.optimizers.AdamW(
                    learning_rate=WARMUP_LR,
                    weight_decay=1e-4,
                    clipnorm=1.0
                )
                model.compile(optimizer=warmup_optimizer, **compile_kwargs)

                warmup_callbacks = [
                    tf.keras.callbacks.CSVLogger(filename=str(fold_warmup_log), separator=',', append=True),
                    tf.keras.callbacks.EarlyStopping(
                        monitor='val_species_head_venomous_penalty_metric',
                        mode='min',
                        patience=2,
                        restore_best_weights=True,
                        verbose=1
                    )
                ]
                model.fit(train_ds, validation_data=val_ds, epochs=WARMUP_EPOCHS, callbacks=warmup_callbacks)
                
                model.save(str(fold_warmup_model_path))
                print(f"Warm-up modell sikeresen elmentve: {fold_warmup_model_path}")

            print(f"\n=== 2. FÁZIS: ResNet finomhangolása (Fold {fold}, {FINETUNE_EPOCHS} epoch) ===")
            unfreeze_resnet_top(backbone, num_layers=30)
            model.compile(optimizer=Adam(learning_rate=FINETUNE_LR), **compile_kwargs)

            history = model.fit(
                train_ds,
                validation_data=val_ds,
                epochs=WARMUP_EPOCHS + FINETUNE_EPOCHS,
                initial_epoch=WARMUP_EPOCHS,
                callbacks=finetune_callbacks
            )

        print(f"\n[Fold {fold} KÉSZ] A legjobb modell elmentve ide: {fold_model_path}")

        if history is not None:
            plot_training_history(history, str(fold_plot_path), fold)
            
        # Memória felszabadítása a következő iteráció előtt
        tf.keras.backend.clear_session()