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
    ZIP_PATH = DRIVE_DIR / 'train_images_224_letterboxed.zip'
    EXTRACT_PATH = Path('/content/train_images_224_letterboxed')

    if not EXTRACT_PATH.exists() and ZIP_PATH.exists():
        print(f"Képek kicsomagolása a helyi SSD-re: {EXTRACT_PATH} ...")
        with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
            zip_ref.extractall(EXTRACT_PATH)
        print("Kicsomagolás kész!")

    LOCAL_DATA_DIR = Path('/content/data_local')
    LOCAL_DATA_DIR.mkdir(parents=True, exist_ok=True)
    for csv_name in ["train_split_fixed.csv", "val_split_fixed.csv"]:
        src_csv = DRIVE_DIR / csv_name
        dst_csv = LOCAL_DATA_DIR / csv_name
        if src_csv.exists() and not dst_csv.exists():
            shutil.copy2(src_csv, dst_csv)

    DATA_DIR = LOCAL_DATA_DIR if (LOCAL_DATA_DIR / "train_split_fixed.csv").exists() else DRIVE_DIR
    nested_dir = EXTRACT_PATH / "train_images_224_letterboxed"
    IMAGE_DIR = nested_dir if nested_dir.exists() else EXTRACT_PATH
    MODEL_PATH = DRIVE_DIR / "snake_multitask_model.keras"
    WARMUP_LOG_PATH = DRIVE_DIR / "warmup_log.csv"
    PLOT_PATH = DRIVE_DIR / "lr_finder_plot.png"
else:
    try:
        PROJECT_ROOT = Path(__file__).resolve().parents[1]
    except NameError:
        PROJECT_ROOT = Path().resolve().parents[1]
    DATA_DIR = PROJECT_ROOT / "data"
    IMAGE_DIR = DATA_DIR / "train_images_224_letterboxed"
    MODEL_PATH = PROJECT_ROOT / "models" / "snake_multitask_model.keras"
    WARMUP_LOG_PATH = PROJECT_ROOT / "models" / "warmup_log.csv"
    PLOT_PATH = PROJECT_ROOT / "models" / "lr_finder_plot.png"
    os.makedirs(PROJECT_ROOT / "models", exist_ok=True)

# Hiperparaméterek
BATCH_SIZE = 128
WARMUP_EPOCHS = 3
FINETUNE_EPOCHS = 10
DEFAULT_WARMUP_LR = 2e-3
FINETUNE_LR = 1.5e-4
RUN_LR_FINDER = True  # Ha True, az 1. fázis előtt automatikusan megkeresi a legjobb WARMUP_LR-t

# ---------------------------------------------------------
# 2. GYORS ADATBETÖLTŐ PIPELINE (uint8 + INTEGER_FAST)
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
# 4. AUTOMATIKUS LEARNING RATE FINDER CALLBACK
# ---------------------------------------------------------
class LRFinder(tf.keras.callbacks.Callback):
    def __init__(self, start_lr=1e-6, end_lr=1e-1, max_steps=200):
        super().__init__()
        self.start_lr = start_lr
        self.end_lr = end_lr
        self.max_steps = max_steps
        self.factor = (end_lr / start_lr) ** (1.0 / max_steps)
        self.lrs = []
        self.losses = []
        self.smoothed_losses = []
        self.best_loss = float('inf')
        self.step = 0

    def _set_lr(self, lr):
        # Kompatibilis mind a régebbi TF 2.x, mind az új Keras 3 optimalizálókkal
        if hasattr(self.model.optimizer, 'learning_rate'):
            self.model.optimizer.learning_rate.assign(lr)
        else:
            tf.keras.backend.set_value(self.model.optimizer.lr, lr)

    def on_train_batch_begin(self, batch, logs=None):
        lr = self.start_lr * (self.factor ** self.step)
        self._set_lr(lr)
        self.lrs.append(lr)

    def on_train_batch_end(self, batch, logs=None):
        loss = float(logs.get('loss', 0.0))
        self.losses.append(loss)

        if self.step == 0:
            smooth = loss
        else:
            smooth = 0.85 * self.smoothed_losses[-1] + 0.15 * loss
        self.smoothed_losses.append(smooth)

        if smooth < self.best_loss:
            self.best_loss = smooth

        self.step += 1

        # Ha a hiba felrobban (4x nagyobb, mint az eddigi legjobb), azonnal leállítjuk a tesztet
        if self.step > 20 and (np.isnan(loss) or smooth > 4.0 * self.best_loss):
            print(f"\n[LRFinder] Hiba kilőtt (loss={loss:.3f}) -> teszt leállítása a(z) {self.step}. lépésnél.")
            self.model.stop_training = True

    def get_optimal_lr(self, default_lr=2e-3):
        if len(self.smoothed_losses) < 30:
            return default_lr
        # Levágjuk az első 10 és az utolsó 5 zajos lépést, és megkeressük a legmeredekebb esést
        losses_arr = np.array(self.smoothed_losses[10:-5])
        lrs_arr = np.array(self.lrs[10:-5])
        gradients = np.gradient(losses_arr)
        min_grad_idx = int(np.argmin(gradients))
        optimal_lr = float(lrs_arr[min_grad_idx])
        # Biztonsági korlát: [5e-4, 1e-2] közé szorítjuk
        return float(np.clip(optimal_lr, 5e-4, 1e-2))

    def plot_and_save(self, save_path, suggested_lr):
        plt.figure(figsize=(10, 5))
        plt.plot(self.lrs, self.smoothed_losses, label='Kisimított Loss', color='#2b5c8f', linewidth=2)
        plt.axvline(suggested_lr, color='#d9534f', linestyle='--', linewidth=2,
                    label=f'Ajánlott Warm-up LR: {suggested_lr:.2e}')
        plt.xscale('log')
        plt.xlabel('Tanulási ráta (Log skála)')
        plt.ylabel('Költségérzékeny Loss')
        plt.title('LR Finder eredménye (Legmeredekebb csökkenés)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.show()
        print(f"[LRFinder] Grafikon elmentve ide: {save_path}")

# ---------------------------------------------------------
# 5. TRANSFER LEARNING MODELL (ResNet50)
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

# ---------------------------------------------------------
# 6. FŐPROGRAM (LR Finder -> Warm-up -> Finomhangolás)
# ---------------------------------------------------------
if __name__ == "__main__":
    print(f"Adatok betöltése innen: {DATA_DIR} | Képek innen: {IMAGE_DIR}")
    train_ds = create_dataset(DATA_DIR / "train_split_fixed.csv", IMAGE_DIR, BATCH_SIZE, is_training=True)
    val_ds = create_dataset(DATA_DIR / "val_split_fixed.csv", IMAGE_DIR, BATCH_SIZE, is_training=False)

    compile_kwargs = {
        'loss': {'species_head': cost_sensitive_species_loss, 'venomous_head': weighted_binary_venomous_loss},
        'loss_weights': {'species_head': 1.0, 'venomous_head': 0.2},
        'metrics': {'species_head': ['accuracy', venomous_penalty_metric], 'venomous_head': 'accuracy'},
        'jit_compile': False
    }

    finetune_callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(MODEL_PATH),
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
            verbose=1
        )
    ]

    # --- A) HA MÁR VAN ELMENTETT MODELL: FOLYTATÁS ---
    if MODEL_PATH.exists():
        print(f"\n[+] Meglévő modell betöltése innen: {MODEL_PATH}")
        model = tf.keras.models.load_model(str(MODEL_PATH))
        backbone = model.get_layer('resnet50')

        print(f"\n[!] Az LR Findert és az 1. FÁZIST (Warm-up) átugorjuk.")
        print(f"2. FÁZIS: Finomhangolás FOLYTATÁSA ({FINETUNE_EPOCHS} epoch, Batch={BATCH_SIZE}, LR=1e-4)...")

        unfreeze_resnet_top(backbone, num_layers=30)
        model.compile(optimizer=Adam(learning_rate=1e-4), **compile_kwargs)
        model.fit(train_ds, validation_data=val_ds, epochs=FINETUNE_EPOCHS, callbacks=finetune_callbacks)

    # --- B) HA MÉG NINCS MODELL: LR FINDER + WARMUP + FINETUNE ---
    else:
        chosen_warmup_lr = DEFAULT_WARMUP_LR

        # 0. LÉPÉS: LR Finder futtatása (200 lépésen = 25 600 képen)
        if RUN_LR_FINDER:
            print("\n=== 0. FÁZIS: LR Finder futtatása (200 lépés a legjobb Warm-up LR megtalálásához) ===")
            temp_model, _ = build_model()
            temp_model.compile(optimizer=Adam(learning_rate=1e-6), **compile_kwargs)

            lr_finder = LRFinder(start_lr=1e-6, end_lr=1e-1, max_steps=200)
            temp_model.fit(train_ds, epochs=1, steps_per_epoch=200, callbacks=[lr_finder], verbose=1)

            chosen_warmup_lr = lr_finder.get_optimal_lr(default_lr=DEFAULT_WARMUP_LR)
            print(f"\n[LRFinder] Megtalált optimális Warm-up LR: {chosen_warmup_lr:.2e}")
            lr_finder.plot_and_save(PLOT_PATH, chosen_warmup_lr)

            # Töröljük a tesztmodellt a memóriából, hogy tiszta súlyokkal induljon a valódi tanítás
            del temp_model
            tf.keras.backend.clear_session()

        # Új, tiszta modell építése az éles tanításhoz
        model, backbone = build_model()

        # 1. FÁZIS: Fej bemelegítése (Warm-up) + CSV naplózás CSAK a Warm-uphoz
        print(f"\n=== 1. FÁZIS: Új fej bemelegítése ({WARMUP_EPOCHS} epoch, Batch={BATCH_SIZE}, LR={chosen_warmup_lr:.2e}) ===")
        warmup_optimizer = tf.keras.optimizers.AdamW(
            learning_rate=chosen_warmup_lr,
            weight_decay=1e-4,
            clipnorm=1.0
        )
        model.compile(optimizer=warmup_optimizer, **compile_kwargs)

        warmup_callbacks = [
            tf.keras.callbacks.CSVLogger(filename=str(WARMUP_LOG_PATH), separator=',', append=True)
        ]
        model.fit(train_ds, validation_data=val_ds, epochs=WARMUP_EPOCHS, callbacks=warmup_callbacks)
        print(f"Warm-up log elmentve ide: {WARMUP_LOG_PATH}")

        # 2. FÁZIS: ResNet felső 30 rétegének finomhangolása
        print(f"\n=== 2. FÁZIS: ResNet finomhangolása ({FINETUNE_EPOCHS} epoch, Batch={BATCH_SIZE}, LR={FINETUNE_LR:.2e}) ===")
        unfreeze_resnet_top(backbone, num_layers=30)
        model.compile(optimizer=Adam(learning_rate=FINETUNE_LR), **compile_kwargs)

        model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=WARMUP_EPOCHS + FINETUNE_EPOCHS,
            initial_epoch=WARMUP_EPOCHS,
            callbacks=finetune_callbacks
        )

    print("\nTanítás befejeződött, a legjobb büntetésű modell elmentve ide:", MODEL_PATH)