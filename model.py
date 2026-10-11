import os
import tensorflow as tf
from tensorflow.keras import layers, models, Input
import pandas as pd

# ---------------------------------------------------------
# BEÁLLÍTÁSOK (Ezt írjátok át a saját mappátokra!)
# ---------------------------------------------------------
IMAGE_DIR = 'train_images_small\\train_images_small' # A mappa, ahol a tényleges .jpg képek vannak
BATCH_SIZE = 32
EPOCHS = 1

# Felesleges TF logok elrejtése
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# ---------------------------------------------------------
# 1. ADATBETÖLTŐ FÜGGVÉNY
# ---------------------------------------------------------
def process_image(file_path, class_id, venomous):
    img = tf.io.read_file(file_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [224, 224])
    img = img / 255.0 # Normalizálás 0-1 közé
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
    # Ha y_true == 1 (mérges), 5.0-s szorzót kap, különben 1.0-t
    weights = tf.where(tf.equal(y_true, 1.0), 5.0, 1.0)
    bce = tf.keras.backend.binary_crossentropy(y_true, y_pred)
    return tf.reduce_mean(bce * tf.squeeze(weights))

# ---------------------------------------------------------
# 3. SAJÁT MULTI-TASK CNN MODELL
# ---------------------------------------------------------
def build_model(num_species=296):
    inputs = Input(shape=(224, 224, 3))
    
    # Közös törzs (Feature extraction)
    x = layers.Conv2D(32, (3, 3), activation='relu')(inputs)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(128, (3, 3), activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    
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
    train_ds = create_dataset("train_split_fixed.csv", IMAGE_DIR, BATCH_SIZE)
    val_ds = create_dataset("val_split_fixed.csv", IMAGE_DIR, BATCH_SIZE)
    
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
    
    print("\nTanítás indítása (Epochs: 10)...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS
    )
    
    # Modell mentése a tanítás végén
    model.save('snake_multitask_model.h5')
    print("Tanítás befejeződött, modell elmentve!")