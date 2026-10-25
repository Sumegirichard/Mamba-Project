import os
import shutil
import zipfile
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, Input, mixed_precision
from tensorflow.keras.applications import EfficientNetV2B2
from tensorflow.keras.optimizers import Adam

# Véletlenszám-generátorok fixálása a reprodukálhatóságért
tf.random.set_seed(41)
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
    DRIVE_DIR = Path('/content/drive/MyDrive/Mamba/')
    
    ZIP_PATH = DRIVE_DIR / 'Databank/Train data/train_images_224_letterboxed.zip'
    EXTRACT_PATH = Path('/content/Databank/Train data')
    CSV_SOURCE_DIR = DRIVE_DIR / 'CSV files' / 'folds'
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
            print("A képek már ki vannak csomagolva a helyi SSD-re.")
    
    LOCAL_DATA_DIR = Path('/content/data_local/folds')
    LOCAL_DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    csv_name = "fold_1_train.csv"
    src_csv = CSV_SOURCE_DIR / csv_name
    dst_csv = LOCAL_DATA_DIR / csv_name
    if src_csv.exists() and not dst_csv.exists():
        shutil.copy2(src_csv, dst_csv)

    DATA_DIR = LOCAL_DATA_DIR if (LOCAL_DATA_DIR / csv_name).exists() else CSV_SOURCE_DIR
    
    if (EXTRACT_PATH / "train_images_224_letterboxed").exists():
        IMAGE_DIR = EXTRACT_PATH / "train_images_224_letterboxed"
    else:
        IMAGE_DIR = EXTRACT_PATH
        
    PLOT_PATH = LOGS_DIR / "lr_finder_effnet_warmup_plot.png"
else:
    try:
        PROJECT_ROOT = Path(__file__).resolve().parents[1]
    except NameError:
        PROJECT_ROOT = Path().resolve().parents[1]
    DATA_DIR = PROJECT_ROOT / "data" / "CSV files" / "folds"
    IMAGE_DIR = PROJECT_ROOT / "data" / "Databank" / "Train data" / "train_images_224_letterboxed"
    PLOT_PATH = PROJECT_ROOT / "models" / "lr_finder_effnet_warmup_plot.png"
    os.makedirs(PROJECT_ROOT / "models", exist_ok=True)

# Hiperparaméterek
BATCH_SIZE = 128
DEFAULT_WARMUP_LR = 5e-4

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

# ---------------------------------------------------------
# 4. ÚJ, PONTOSABB LEARNING RATE FINDER
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

        if self.step > 20 and (np.isnan(loss) or smooth > 4.0 * self.best_loss):
            print(f"\n[LRFinder] Hiba kilőtt (loss={loss:.3f}) -> teszt leállítása a(z) {self.step}. lépésnél.")
            self.model.stop_training = True

    def get_optimal_lr(self, default_lr=5e-4):
        if len(self.smoothed_losses) < 50:
            return default_lr
            
        skip_begin = 30
        skip_end = 5
        
        losses_arr = np.array(self.smoothed_losses[skip_begin:-skip_end])
        lrs_arr = np.array(self.lrs[skip_begin:-skip_end])
        
        if len(losses_arr) < 10:
            return default_lr
            
        gradients = np.gradient(losses_arr)
        smoothed_gradients = np.convolve(gradients, np.ones(10)/10, mode='valid')
        
        pad = 10 // 2
        min_grad_idx = int(np.argmin(smoothed_gradients)) + pad
        
        optimal_lr = float(lrs_arr[min_grad_idx])
        return float(np.clip(optimal_lr, 1e-5, 1e-2))

    def plot_and_save(self, save_path, suggested_lr):
        plt.figure(figsize=(10, 5))
        plt.plot(self.lrs, self.smoothed_losses, label='Kisimított Loss', color='#2b5c8f', linewidth=2)
        plt.axvline(suggested_lr, color='#d9534f', linestyle='--', linewidth=2,
                    label=f'Ajánlott Warm-up LR: {suggested_lr:.2e}')
        plt.xscale('log')
        plt.xlabel('Tanulási ráta (Log skála)')
        plt.ylabel('Költségérzékeny Loss')
        plt.title('LR Finder eredménye (EfficientNetV2B2 - Warm-up)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.show()
        print(f"[LRFinder] Grafikon elmentve ide: {save_path}")

# ---------------------------------------------------------
# 5. TRANSFER LEARNING MODELL (EfficientNetV2B2)
# ---------------------------------------------------------
def build_model(num_species=296):
    inputs = Input(shape=(224, 224, 3), dtype=tf.uint8)
    
    x = layers.Rescaling(scale=1.0, dtype='float32')(inputs)
    x = data_augmentation(x)

    backbone = EfficientNetV2B2(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
    backbone.trainable = False

    x = layers.GlobalAveragePooling2D()(backbone(x, training=False))
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.3)(x)

    species_out = layers.Dense(num_species, activation='softmax', dtype='float32', name='species_head')(x)
    venomous_out = layers.Dense(1, activation='sigmoid', dtype='float32', name='venomous_head')(x)

    return models.Model(inputs=inputs, outputs=[species_out, venomous_out])

# ---------------------------------------------------------
# 6. FŐPROGRAM (LR Finder futtatása)
# ---------------------------------------------------------
if __name__ == "__main__":
    print(f"Adatok betöltése innen: {DATA_DIR} | Képek innen: {IMAGE_DIR}")
    train_ds = create_dataset(DATA_DIR / "fold_1_train.csv", IMAGE_DIR, BATCH_SIZE, is_training=True)

    compile_kwargs = {
        'loss': {'species_head': cost_sensitive_species_loss, 'venomous_head': weighted_binary_venomous_loss},
        'loss_weights': {'species_head': 1.0, 'venomous_head': 0.2},
        'metrics': {'species_head': ['accuracy'], 'venomous_head': ['accuracy']},
        'jit_compile': False
    }

    print("\n=== LR Finder futtatása a Warm-up fázishoz (200 lépés, 1e-6 -> 1e-1) ===")
    temp_model = build_model()
    temp_model.compile(optimizer=Adam(learning_rate=1e-6), **compile_kwargs)

    lr_finder = LRFinder(start_lr=1e-6, end_lr=1e-1, max_steps=200)
    temp_model.fit(train_ds, epochs=1, steps_per_epoch=200, callbacks=[lr_finder], verbose=1)

    chosen_warmup_lr = lr_finder.get_optimal_lr(default_lr=DEFAULT_WARMUP_LR)
    print(f"\n[LRFinder] Megtalált optimális Warm-up LR: {chosen_warmup_lr:.2e}")
    print(f"--> Ezt az értéket ({chosen_warmup_lr:.6f}) írd be a model_EfficientNet.py WARMUP_LR változójába!")
    lr_finder.plot_and_save(PLOT_PATH, chosen_warmup_lr)