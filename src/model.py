import os
from pathlib import Path
import tensorflow as tf
from tensorflow.keras import layers, models, Input
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.optimizers import Adam
import pandas as pd

# ---------------------------------------------------------
# BEÁLLÍTÁSOK
# ---------------------------------------------------------
try:
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
except NameError:
    PROJECT_ROOT = Path().resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
IMAGE_DIR = DATA_DIR / "train_images_224_letterboxed"
MODEL_PATH = PROJECT_ROOT / "models" / "snake_multitask_model.keras"

BATCH_SIZE = 32
EPOCHS = 10 

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# ---------------------------------------------------------
# 1. ADATBETÖLTŐ ÉS AUGMENTÁCIÓS FÜGGVÉNYEK
# ---------------------------------------------------------
def process_image(file_path, class_id, venomous):
    img = tf.io.read_file(file_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.cast(img, tf.float32)
    img = preprocess_input(img)
    return img, {'species_head': class_id, 'venomous_head': venomous}

# Az augmentáció definíciója megmarad globálisan
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.15),
    layers.RandomZoom(0.1),
])

# KIVETTÜK az is_training logikát, mert ezt most a modell maga kezeli a GPU-n
def create_dataset(csv_file, img_dir, batch_size):
    df = pd.read_csv(csv_file)
    file_paths = [os.path.join(img_dir, str(p)) for p in df['image_path']]
    
    ds = tf.data.Dataset.from_tensor_slices((file_paths, df['class_id'].values, df['MIVS'].values))
    ds = ds.map(process_image, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.shuffle(buffer_size=1000).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds

# ---------------------------------------------------------
# 2. EGYEDI LOSS FÜGGVÉNY
# ---------------------------------------------------------
def weighted_venomous_loss(y_true, y_pred):
    y_true = tf.cast(tf.reshape(y_true, [-1, 1]), tf.float32)
    y_pred = tf.reshape(y_pred, [-1, 1])
    
    weights = tf.where(tf.equal(y_true, 1.0), 5.0, 1.0)
    
    bce = tf.keras.backend.binary_crossentropy(y_true, y_pred)
    bce = tf.reshape(bce, [-1, 1])
    
    return tf.reduce_mean(bce * weights)

# ---------------------------------------------------------
# 3. TRANSFER LEARNING MODELL (ResNet50)
# ---------------------------------------------------------
def build_model(num_species=296):
    inputs = Input(shape=(224, 224, 3))
    
    # 1. VÁLTOZÁS: Az augmentációs réteg ide kerül, így a GPU végzi a számítást
    x = data_augmentation(inputs)
    
    # 2. VÁLTOZÁS: A gerinc bemenete immár az augmentált tenzor (x) lesz
    backbone = ResNet50(input_tensor=x, include_top=False, weights='imagenet')
    
    backbone.trainable = True

    for layer in backbone.layers[:-20]:
        layer.trainable = False 
    
    x = backbone.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    
    species_out = layers.Dense(num_species, activation='softmax', name='species_head')(x)
    venomous_out = layers.Dense(1, activation='sigmoid', name='venomous_head')(x)
    
    return models.Model(inputs=inputs, outputs=[species_out, venomous_out])

# ---------------------------------------------------------
# 4. FŐPROGRAM (Futtatás)
# ---------------------------------------------------------
if __name__ == "__main__":
    print("Adatok betöltése...")
    
    # KIVETTÜK az is_training paramétereket a meghívásból
    train_ds = create_dataset(DATA_DIR / "train_split_fixed.csv", IMAGE_DIR, BATCH_SIZE)
    val_ds = create_dataset(DATA_DIR / "val_split_fixed.csv", IMAGE_DIR, BATCH_SIZE)
    
    model = build_model()
    
    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss={
            'species_head': 'sparse_categorical_crossentropy',
            'venomous_head': weighted_venomous_loss
        },
        loss_weights={
            'species_head': 1.0,
            'venomous_head': 0.2
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
    
    os.makedirs(PROJECT_ROOT / "models", exist_ok=True)
    model.save(str(MODEL_PATH))
    print("\nTanítás befejeződött, modell elmentve ide:", MODEL_PATH)