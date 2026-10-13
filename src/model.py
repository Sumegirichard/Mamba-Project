import os
from pathlib import Path
import tensorflow as tf
from tensorflow.keras import layers, models, Input
from tensorflow.keras.applications import EfficientNetB2 # 1. VÁLTOZÁS: Új import
import pandas as pd

# ---------------------------------------------------------
# BEÁLLÍTÁSOK
# ---------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
IMAGE_DIR = DATA_DIR / "train_images_small" / "train_images_small"
MODEL_PATH = PROJECT_ROOT / "models" / "snake_multitask_model.h5"
BATCH_SIZE = 32
EPOCHS = 1 # 2. VÁLTOZÁS: 1-ről 10-re emelve, hogy legyen ideje tanulni

# Felesleges TF logok elrejtése
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# ---------------------------------------------------------
# 1. ADATBETÖLTŐ FÜGGVÉNY
# ---------------------------------------------------------
def process_image(file_path, class_id, venomous):
    img = tf.io.read_file(file_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [224, 224])
    
    # 3. VÁLTOZÁS: Az EfficientNet maga végzi a normalizálást!
    # A / 255.0 sort kikommentezve hagytuk, hogy a hálózat az eredeti értékeket kapja.
    # img = img / 255.0 
    
    return img, {'species_head': class_id, 'venomous_head': venomous}

def create_dataset(csv_file, img_dir, batch_size):
    df = pd.read_csv(csv_file)
    file_paths = [os.path.join(img_dir, str(p)) for p in df['image_path']]
    
    ds = tf.data.Dataset.from_tensor_slices((file_paths, df['class_id'].values, df['MIVS'].values))
    ds = ds.map(process_image, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.shuffle(buffer_size=1000).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds

# ---------------------------------------------------------
# 2. EGYEDI LOSS FÜGGVÉNY (5x büntetés a mérges elvétéséért)
# ---------------------------------------------------------
def weighted_venomous_loss(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    weights = tf.where(tf.equal(y_true, 1.0), 5.0, 1.0)
    bce = tf.keras.backend.binary_crossentropy(y_true, y_pred)
    return tf.reduce_mean(bce * tf.squeeze(weights))

# ---------------------------------------------------------
# 3. TRANSFER LEARNING MODELL (EfficientNetB2)
# ---------------------------------------------------------
def build_model(num_species=296):
    inputs = Input(shape=(224, 224, 3))
    
    # 4. VÁLTOZÁS: Saját CNN helyett az előre betanított EfficientNet gerinc
    backbone = EfficientNetB2(input_tensor=inputs, include_top=False, weights='imagenet')
    
    # Befagyasztjuk az eredeti súlyokat, hogy csak a kígyós fejek tanuljanak
    backbone.trainable = False 
    
    x = backbone.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.3)(x) # 30% felejtés túltanulás ellen
    
    # Fej 1: Faj felismerése
    species_out = layers.Dense(num_species, activation='softmax', name='species_head')(x)
    
    # Fej 2: Mérgezőség felismerése
    venomous_out = layers.Dense(1, activation='sigmoid', name='venomous_head')(x)
    
    return models.Model(inputs=inputs, outputs=[species_out, venomous_out])

# ---------------------------------------------------------
# 4. FŐPROGRAM (Futtatás)
# ---------------------------------------------------------
if __name__ == "__main__":
    print("Adatok betöltése...")
    train_ds = create_dataset(DATA_DIR / "train_split_fixed.csv", IMAGE_DIR, BATCH_SIZE)
    val_ds = create_dataset(DATA_DIR / "val_split_fixed.csv", IMAGE_DIR, BATCH_SIZE)
    
    model = build_model()
    
    model.compile(
        optimizer='adam',
        loss={
            'species_head': 'sparse_categorical_crossentropy',
            'venomous_head': weighted_venomous_loss
        },
        metrics={
            'species_head': 'accuracy',
            'venomous_head': 'accuracy'
        }
    )
    
    print(f"\nTanítás indítása (Epochs: {EPOCHS})...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS
    )
    
    # Modell mentése a tanítás végén
    os.makedirs(PROJECT_ROOT / "models", exist_ok=True)
    model.save(str(MODEL_PATH))
    print("Tanítás befejeződött, modell elmentve ide:", MODEL_PATH)