import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import json
import argparse
import shutil
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import layers, models, Input, mixed_precision
from tensorflow.keras.optimizers import AdamW
from transformers import TFSwinModel

# ---------------------------------------------------------
# 0. REPRODUKÁLHATÓSÁG ÉS MIXED PRECISION
# ---------------------------------------------------------
tf.random.set_seed(42)
np.random.seed(42)

try:
    register_serializable = tf.keras.utils.register_keras_serializable
except AttributeError:
    from keras.saving import register_keras_serializable as register_serializable

mixed_precision.set_global_policy("mixed_float16")
print("Aktív számítási pontosság:", mixed_precision.global_policy().name)

# ---------------------------------------------------------
# 1. KÖRNYEZET ÉS ADATÚTVONALAK
# ---------------------------------------------------------
IN_COLAB = False

try:
    from google.colab import drive
    IN_COLAB = True
except ImportError:
    pass

if IN_COLAB:
    drive.mount("/content/drive")
    DRIVE_DIR = Path("/content/drive/MyDrive/Mamba")

    ZIP_PATH = DRIVE_DIR / "Databank/Train data/train_images_384_letterboxed.zip"
    EXTRACT_PATH = Path("/content/Databank/Train data")
    FOLDS_SOURCE_DIR = DRIVE_DIR / "CSV files" / "folds"
    LOGS_DIR = DRIVE_DIR / "Logs"

    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    if ZIP_PATH.exists():
        if not (EXTRACT_PATH / "train_images_384_letterboxed").exists():
            print(f"Képek kicsomagolása a helyi SSD-re: {EXTRACT_PATH} ...")
            EXTRACT_PATH.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(ZIP_PATH, "r") as zip_ref:
                zip_ref.extractall(EXTRACT_PATH)
            print("Kicsomagolás kész!")
    else:
        print(f"FIGYELEM: A ZIP nem található: {ZIP_PATH}")

    LOCAL_FOLDS_DIR = Path("/content/data_local/folds")
    LOCAL_FOLDS_DIR.mkdir(parents=True, exist_ok=True)

    if FOLDS_SOURCE_DIR.exists():
        for i in range(1, 6):
            for split in ["train", "val"]:
                file_name = f"fold_{i}_{split}.csv"
                src_csv = FOLDS_SOURCE_DIR / file_name
                dst_csv = LOCAL_FOLDS_DIR / file_name
                if src_csv.exists() and not dst_csv.exists():
                    shutil.copy2(src_csv, dst_csv)

    DATA_DIR = LOCAL_FOLDS_DIR if (LOCAL_FOLDS_DIR / "fold_1_train.csv").exists() else FOLDS_SOURCE_DIR
    IMAGE_DIR = EXTRACT_PATH / "train_images_384_letterboxed" if (EXTRACT_PATH / "train_images_384_letterboxed").exists() else EXTRACT_PATH

else:
    PROJECT_ROOT = Path().resolve().parents[1]
    DATA_DIR = PROJECT_ROOT / "data" / "CSV files" / "folds"
    IMAGE_DIR = PROJECT_ROOT / "data" / "Databank" / "Train data" / "train_images_384_letterboxed"
    DRIVE_DIR = PROJECT_ROOT / "models"
    LOGS_DIR = PROJECT_ROOT / "models"
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------
# 2. HIPERPARAMÉTEREK (384x384 Progressive Resizing)
# ---------------------------------------------------------
IMAGE_SIZE = 384
BATCH_SIZE = 32

FINETUNE_EPOCHS = 20
FINETUNE_LR = 5e-5
WEIGHT_DECAY = 1e-4

NUM_CLASSES = 296
NUM_FOLDS = 5
FOLDS_TO_RUN = [1]

EARLY_STOPPING_PATIENCE = 5
EARLY_STOPPING_MIN_DELTA = 1e-3

# ---------------------------------------------------------
# 3. ADATBETÖLTŐ PIPELINE
# ---------------------------------------------------------
def process_image(file_path, class_id, venomous):
    img = tf.io.read_file(file_path)
    img = tf.io.decode_jpeg(img, channels=3, dct_method="INTEGER_FAST")
    img = tf.ensure_shape(img, [IMAGE_SIZE, IMAGE_SIZE, 3])
    return img, {"species_head": class_id, "venomous_head": venomous}

data_augmentation = tf.keras.Sequential(
    [
        layers.RandomFlip("horizontal", dtype="float32"),
        layers.RandomRotation(0.15, fill_mode="reflect", dtype="float32"),
        layers.RandomZoom(0.1, fill_mode="reflect", dtype="float32"),
    ],
    name="gpu_augmentation"
)

def create_dataset(csv_file, img_dir, batch_size, is_training=True):
    df = pd.read_csv(csv_file)
    img_dir_str = str(img_dir)
    file_paths = [os.path.join(img_dir_str, str(p)) for p in df["image_path"]]

    ds = tf.data.Dataset.from_tensor_slices(
        (
            file_paths,
            df["class_id"].values.astype("int32"),
            df["MIVS"].values.astype("float32")
        )
    )

    options = tf.data.Options()
    options.experimental_deterministic = not is_training
    ds = ds.with_options(options)

    if is_training:
        ds = ds.shuffle(buffer_size=len(df), reshuffle_each_iteration=True)

    ds = ds.map(process_image, num_parallel_calls=tf.data.AUTOTUNE)

    if not is_training:
        ds = ds.cache()

    return ds.batch(batch_size, drop_remainder=is_training).prefetch(tf.data.AUTOTUNE)

# ---------------------------------------------------------
# 4. MÉRGEK ÉS EGYEDI LOSS-FÜGGVÉNYEK
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

if int(tf.size(v_mapping).numpy()) != NUM_CLASSES:
    raise ValueError(f"A v_mapping hossza {int(tf.size(v_mapping).numpy())}, de NUM_CLASSES={NUM_CLASSES}.")

v_mapping_float = tf.cast(v_mapping, tf.float32)

@register_serializable()
class SparseMacroF1(tf.keras.metrics.Metric):
    def __init__(self, name="macro_f1", num_classes=296, **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.tp = self.add_weight(name="tp", shape=(num_classes,), initializer="zeros", dtype=tf.float32)
        self.fp = self.add_weight(name="fp", shape=(num_classes,), initializer="zeros", dtype=tf.float32)
        self.fn = self.add_weight(name="fn", shape=(num_classes,), initializer="zeros", dtype=tf.float32)

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
        y_pred = tf.cast(tf.argmax(y_pred, axis=-1), tf.int32)

        y_true_1hot = tf.one_hot(y_true, depth=self.num_classes, dtype=tf.float32)
        y_pred_1hot = tf.one_hot(y_pred, depth=self.num_classes, dtype=tf.float32)

        self.tp.assign_add(tf.reduce_sum(y_true_1hot * y_pred_1hot, axis=0))
        self.fp.assign_add(tf.reduce_sum((1.0 - y_true_1hot) * y_pred_1hot, axis=0))
        self.fn.assign_add(tf.reduce_sum(y_true_1hot * (1.0 - y_pred_1hot), axis=0))

    def result(self):
        epsilon = tf.keras.backend.epsilon()
        precision = self.tp / (self.tp + self.fp + epsilon)
        recall = self.tp / (self.tp + self.fn + epsilon)
        return tf.reduce_mean(2.0 * precision * recall / (precision + recall + epsilon))

    def reset_state(self):
        self.tp.assign(tf.zeros(self.num_classes, dtype=tf.float32))
        self.fp.assign(tf.zeros(self.num_classes, dtype=tf.float32))
        self.fn.assign(tf.zeros(self.num_classes, dtype=tf.float32))

    def get_config(self):
        config = super().get_config()
        config.update({"num_classes": self.num_classes})
        return config

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
# 5. SWIN TRANSFORMER MODELL 384x384 BEMENETTEL
# ---------------------------------------------------------
def build_model_384(num_species=NUM_CLASSES):
    inputs = Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3), dtype=tf.uint8)

    x = data_augmentation(tf.cast(inputs, tf.float32))

    # Swin / HuggingFace specifikus 0-1 skálázás és ImageNet normalizálás 384-re
    x = x / 255.0
    mean = tf.constant([0.485, 0.456, 0.406], shape=[1, 1, 1, 3], dtype=tf.float32)
    std = tf.constant([0.229, 0.224, 0.225], shape=[1, 1, 1, 3], dtype=tf.float32)
    x = (x - mean) / std

    print("\n[+] Hugging Face Swin Transformer (384x384, window12) létrehozása...")
    backbone = TFSwinModel.from_pretrained("microsoft/swin-base-patch4-window12-384", from_pt=True)
    backbone.trainable = True

    outputs = backbone(pixel_values=x, training=False)
    x = outputs.pooler_output

    x = layers.Dense(512, activation="gelu")(x)
    x = layers.Dropout(0.3)(x)

    s_out = layers.Dense(num_species, activation="softmax", dtype="float32", name="species_head")(x)
    v_out = layers.Dense(1, activation="sigmoid", dtype="float32", name="venomous_head")(x)

    model = models.Model(inputs=inputs, outputs=[s_out, v_out])
    return model, backbone

# ---------------------------------------------------------
# 6. GRAFIKON ÉS EREDMÉNYEK MENTÉSE
# ---------------------------------------------------------
def plot_training_history(history, save_path, fold):
    if history is None:
        return
    hist = history.history
    plt.figure(figsize=(10, 6))
    plt.plot(hist.get("loss", []), label="Tanítási Loss", linewidth=2)
    plt.plot(hist.get("val_loss", []), label="Validációs Loss", linewidth=2)
    plt.title(f"Swin Transformer 384x384 - Fold {fold}")
    plt.xlabel("Epoch"); plt.ylabel("Teljes Loss"); plt.legend(); plt.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(save_path, dpi=150); plt.close()
    print(f"[+] Loss grafikon mentve: {save_path}")

def save_fold_result(history, result_path, fold):
    if history is None or "val_loss" not in history.history:
        return
    hist = history.history
    best_idx = int(np.argmin(hist["val_loss"]))
    result = {"fold": fold, "image_size": IMAGE_SIZE, "best_epoch_in_this_run": best_idx + 1}
    for key, values in hist.items():
        if key.startswith("val_"):
            result[key] = float(values[best_idx])

    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"[+] Eredmények mentve: {result_path}")

# ---------------------------------------------------------
# 7. EGY FOLD FINOMHANGOLÁSA 224 -> 384
# ---------------------------------------------------------
def train_fold(fold):
    print("\n" + "=" * 65)
    print(f"=== {fold}. FOLD: Swin Transformer 224 -> 384 Progressive Resizing ===")
    print("=" * 65)

    train_csv = DATA_DIR / f"fold_{fold}_train.csv"
    val_csv = DATA_DIR / f"fold_{fold}_val.csv"

    # A 224x224-es Swin modell súlyai
    SWIN_MODELS_DIR = DRIVE_DIR / "models" / "swin" if IN_COLAB else DRIVE_DIR / "swin"
    SWIN_MODELS_DIR.mkdir(parents=True, exist_ok=True)

    PREVIOUS_MODEL_PATH = SWIN_MODELS_DIR / f"snake_swin_model_fold_{fold}.weights.h5"
    fold_model_path = SWIN_MODELS_DIR / f"snake_swin_model_384_fold_{fold}.weights.h5"

    fold_finetune_log = LOGS_DIR / f"swin_384_finetune_log_fold_{fold}.csv"
    fold_plot_path = LOGS_DIR / f"swin_384_training_loss_plot_fold_{fold}.png"
    fold_result_path = LOGS_DIR / f"swin_384_fold_{fold}_result.json"

    for required_file in [train_csv, val_csv]:
        if not required_file.exists():
            print(f"[!] HIÁNYZÓ ADATFÁJL: {required_file}"); return

    if not IMAGE_DIR.exists():
        print(f"[!] A képmappa nem található: {IMAGE_DIR}"); return

    if not PREVIOUS_MODEL_PATH.exists():
        print(f"\n[!] Nem található a 224x224-es kiinduló Swin modell: {PREVIOUS_MODEL_PATH}")
        print("Először futtasd le a 224x224-es Swin tanítást (model_swin_224.py)!")
        return

    if fold_model_path.exists():
        print(f"\n[!] A 384x384-es Swin modell már létezik: {fold_model_path}")
        return

    train_ds = create_dataset(train_csv, IMAGE_DIR, BATCH_SIZE, is_training=True)
    val_ds = create_dataset(val_csv, IMAGE_DIR, BATCH_SIZE, is_training=False)

    compile_kwargs = {
        "loss": {
            "species_head": cost_sensitive_species_loss,
            "venomous_head": weighted_binary_venomous_loss
        },
        "loss_weights": {"species_head": 1.0, "venomous_head": 0.2},
        "metrics": {
            "species_head": ["accuracy", SparseMacroF1(num_classes=NUM_CLASSES), venomous_penalty_metric],
            "venomous_head": ["accuracy"]
        },
        "jit_compile": False
    }

    model, backbone = build_model_384()

    # SÚLYOK BETÖLTÉSE A 224-ES SWIN MODELLBŐL
    print(f"\n[+] A 224x224-es Swin súlyainak betöltése:\n{PREVIOUS_MODEL_PATH}")
    try:
        model.load_weights(str(PREVIOUS_MODEL_PATH), skip_mismatch=True)
        print("[+] A korábbi súlyok (kimeneti fejek és egyező rétegek) sikeresen betöltve.")
    except Exception as exc:
        print(f"\n[!] A súlyok betöltésekor figyelmeztetés/hiba lépett fel: {exc}")

    print(f"\n=== FINOMHANGOLÁS 384x384-EN ({FINETUNE_EPOCHS} epoch) ===")
    backbone.trainable = True

    model.compile(
        optimizer=AdamW(learning_rate=FINETUNE_LR, weight_decay=WEIGHT_DECAY),
        **compile_kwargs
    )

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(fold_model_path), monitor="val_loss", mode="min",
            save_best_only=True, save_weights_only=True, verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", mode="min", factor=0.3, patience=2, min_lr=1e-6,
            min_delta=EARLY_STOPPING_MIN_DELTA, verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", mode="min", patience=EARLY_STOPPING_PATIENCE,
            min_delta=EARLY_STOPPING_MIN_DELTA, restore_best_weights=True, verbose=1
        ),
        tf.keras.callbacks.CSVLogger(str(fold_finetune_log), append=False)
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=FINETUNE_EPOCHS,
        callbacks=callbacks
    )

    model.save_weights(str(fold_model_path))
    print(f"\n[Fold {fold} KÉSZ] A 384x384-es Swin súlyok mentve:\n{fold_model_path}")

    plot_training_history(history, str(fold_plot_path), fold)
    save_fold_result(history, str(fold_result_path), fold)

    del model, backbone, train_ds, val_ds
    tf.keras.backend.clear_session()

# ---------------------------------------------------------
# 8. FŐ CIKLUS
# ---------------------------------------------------------
if __name__ == "__main__":
    print(f"Képek beolvasása innen: {IMAGE_DIR}")
    print(f"CSV fájlok innen: {DATA_DIR}")

    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, nargs="+")
    args, _ = parser.parse_known_args()

    folds = args.fold if args.fold else FOLDS_TO_RUN

    for fold in folds:
        if fold < 1 or fold > NUM_FOLDS:
            raise ValueError(f"Érvénytelen fold: {fold}. Engedélyezett: 1-{NUM_FOLDS}.")

    print(f"Tanítandó fold(ok): {folds}")

    for fold in folds:
        train_fold(fold)

    print(f"\n=== KÉSZ. Lefutott foldok: {folds} ===")