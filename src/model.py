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
WARMUP_EPOCHS = 3    # 1. fázis: csak az új fej bemelegítése
FINETUNE_EPOCHS = 10 # 2. fázis: ResNet felső rétegeinek finomhangolása

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# ---------------------------------------------------------
# 1. ADATBETÖLTŐ ÉS AUGMENTÁCIÓS FÜGGVÉNYEK
# ---------------------------------------------------------
def process_image(file_path, class_id, venomous):
    img = tf.io.read_file(file_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.cast(img, tf.float32)
    img.set_shape([224, 224, 3])
    # FONTOS: A preprocess_input-ot kivettük innen, mert a modellen belül,
    # a GPU-s augmentáció UTÁN kell lefutnia!
    return img, {'species_head': class_id, 'venomous_head': venomous}

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.15, fill_mode="reflect"),
    layers.RandomZoom(0.1, fill_mode="reflect"),
], name="gpu_augmentation")

def create_dataset(csv_file, img_dir, batch_size, is_training=True):
    df = pd.read_csv(csv_file)
    file_paths = [os.path.join(img_dir, str(p)) for p in df['image_path']]
    
    ds = tf.data.Dataset.from_tensor_slices((
        file_paths,
        df['class_id'].values.astype('int32'),
        df['MIVS'].values.astype('float32')
    ))
    
    if is_training:
        # A fájlnév-listát keverjük meg a teljes adathalmazon (memóriabarát!)
        ds = ds.shuffle(buffer_size=len(df), reshuffle_each_iteration=True)
        
    ds = ds.map(process_image, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds

# ---------------------------------------------------------
# 2. EGYEDI LOSS ÉS METRIKA FÜGGVÉNYEK
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

# A) Kiértékelési metrika (a jelenlegi kódodból)
@tf.keras.saving.register_keras_serializable()
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

# B) A prot3.txt-ből átemelt, 4-irányú költségérzékeny faj-loss!
@tf.keras.saving.register_keras_serializable()
def cost_sensitive_species_loss(y_true, y_pred):
    y_true_int = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
    v_y = tf.gather(v_mapping_float, y_true_int)
    
    # Mérges fajokra adott összegzett valószínűség
    v_y_pred = tf.reduce_sum(y_pred * v_mapping_float, axis=-1)
    
    weight_if_venomous = 5.0 * (1.0 - v_y_pred) + 2.0 * v_y_pred
    weight_if_non_venomous = 1.0 * (1.0 - v_y_pred) + 2.0 * v_y_pred
    
    weight = v_y * weight_if_venomous + (1.0 - v_y) * weight_if_non_venomous
    scce = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
    return tf.reduce_mean(scce * weight)

# C) Súlyozott bináris loss a venomous_head-hez (a jelenlegi kódodból)
@tf.keras.saving.register_keras_serializable()
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
    
    # 1. Augmentáció a GPU-n (még a [0, 255] RGB képen!)
    x = data_augmentation(inputs)
    
    # 2. ResNet50 preprocess_input az augmentáció UTÁN
    x = layers.Lambda(preprocess_input, name="resnet_preprocess")(x)
    
    # 3. ResNet50 külön almodellként (kezdetben befagyasztva a Warm-uphoz)
    backbone = ResNet50(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
    backbone.trainable = False
    
    x = backbone(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    
    # 256 helyett 512 neuron, hogy ne legyen szűk keresztmetszet a 296 faj előtt
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    
    species_out = layers.Dense(num_species, activation='softmax', name='species_head')(x)
    venomous_out = layers.Dense(1, activation='sigmoid', name='venomous_head')(x)
    
    return models.Model(inputs=inputs, outputs=[species_out, venomous_out]), backbone

def unfreeze_resnet_top(backbone, num_layers=30):
    """Feloldja az utolsó N réteget, de a BatchNorm rétegeket befagyasztva tartja."""
    backbone.trainable = True
    for layer in backbone.layers[:-num_layers]:
        layer.trainable = False
    for layer in backbone.layers[-num_layers:]:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False
        else:
            layer.trainable = True

# ---------------------------------------------------------
# 4. FŐPROGRAM (Futtatás)
# ---------------------------------------------------------
if __name__ == "__main__":
    print("Adatok betöltése...")
    train_ds = create_dataset(DATA_DIR / "train_split_fixed.csv", IMAGE_DIR, BATCH_SIZE, is_training=True)
    val_ds = create_dataset(DATA_DIR / "val_split_fixed.csv", IMAGE_DIR, BATCH_SIZE, is_training=False)
    
    os.makedirs(PROJECT_ROOT / "models", exist_ok=True)
    model, backbone = build_model()
    
    # 1. FÁZIS: Fej bemelegítése (Warm-up)
    print(f"\n1. FÁZIS: Új fej bemelegítése ({WARMUP_EPOCHS} epoch, LR=1e-3)...")
    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss={'species_head': cost_sensitive_species_loss, 'venomous_head': weighted_binary_venomous_loss},
        loss_weights={'species_head': 1.0, 'venomous_head': 0.2},
        metrics={'species_head': ['accuracy', venomous_penalty_metric], 'venomous_head': 'accuracy'}
    )
    model.fit(train_ds, validation_data=val_ds, epochs=WARMUP_EPOCHS)
    
    # 2. FÁZIS: ResNet finomhangolás
    print(f"\n2. FÁZIS: ResNet finomhangolása ({FINETUNE_EPOCHS} epoch, LR=1e-4)...")
    unfreeze_resnet_top(backbone, num_layers=30)
    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss={'species_head': cost_sensitive_species_loss, 'venomous_head': weighted_binary_venomous_loss},
        loss_weights={'species_head': 1.0, 'venomous_head': 0.2},
        metrics={'species_head': ['accuracy', venomous_penalty_metric], 'venomous_head': 'accuracy'}
    )
    
    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
        filepath=str(MODEL_PATH),
        monitor='val_species_head_venomous_penalty_metric',
        mode='min',
        save_best_only=True,
        verbose=1
    )
    
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=WARMUP_EPOCHS + FINETUNE_EPOCHS,
        initial_epoch=WARMUP_EPOCHS,
        callbacks=[checkpoint_cb]
    )
    
    print("\nTanítás befejeződött, a legjobb büntetésű modell elmentve ide:", MODEL_PATH)