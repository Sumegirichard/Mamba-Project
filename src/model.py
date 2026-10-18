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
# ---------------------------------------------------------
# 2. EGYEDI LOSS ÉS METRIKA FÜGGVÉNYEK
# ---------------------------------------------------------

# A 296 egyedi class_id növekvő sorrendjében a mérgesség (0 vagy 1)
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

# A) A matematikai képlet a fajta (species) kiértékeléséhez (Metrika)
def venomous_penalty_metric(y_true, y_pred):
    y_true_class = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
    y_pred_class = tf.cast(tf.argmax(y_pred, axis=-1), tf.int32)
    
    v_y = tf.gather(v_mapping, y_true_class)
    v_y_hat = tf.gather(v_mapping, y_pred_class)
    
    penalty = tf.zeros_like(tf.cast(y_true_class, tf.float32))
    not_equal = tf.not_equal(y_true_class, y_pred_class)
    
    cond_1 = tf.logical_and(tf.equal(v_y, 0), tf.equal(v_y_hat, 0))
    penalty = tf.where(tf.logical_and(not_equal, cond_1), 1.0, penalty)
    
    cond_2 = tf.logical_and(tf.equal(v_y, 0), tf.equal(v_y_hat, 1))
    penalty = tf.where(tf.logical_and(not_equal, cond_2), 2.0, penalty)
    
    cond_3 = tf.logical_and(tf.equal(v_y, 1), tf.equal(v_y_hat, 1))
    penalty = tf.where(tf.logical_and(not_equal, cond_3), 2.0, penalty)
    
    cond_5 = tf.logical_and(tf.equal(v_y, 1), tf.equal(v_y_hat, 0))
    penalty = tf.where(tf.logical_and(not_equal, cond_5), 5.0, penalty)
    
    return tf.reduce_mean(penalty)

# B) Súlyozott veszteségfüggvény a fajta (species) TANÍTÁSÁHOZ
def weighted_sparse_species_loss(y_true, y_pred):
    y_true_class = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
    v_y = tf.cast(tf.gather(v_mapping, y_true_class), tf.float32)
    
    # 5x-ös súly, ha a valódi fajta mérges
    weights = tf.where(tf.equal(v_y, 1.0), 5.0, 1.0)
    
    # Sparse categorical crossentropy használata az indexek miatt
    scce = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
    return tf.reduce_mean(scce * weights)

# C) A meglévő bináris loss a venomous_head-hez
def weighted_binary_venomous_loss(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    weights = tf.where(tf.equal(y_true, 1.0), 5.0, 1.0)
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    return tf.reduce_mean(bce * tf.reshape(weights, [-1]))

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
            'species_head': weighted_sparse_species_loss,
            'venomous_head': weighted_binary_venomous_loss
        },
        loss_weights={
            'species_head': 1.0,
            'venomous_head': 0.2
        },
        metrics={
            'species_head': ['accuracy', venomous_penalty_metric],
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