import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras import layers, models, Input

# ---------------------------------------------------------
# HELYI ÚTVONALAK (Állítsd be a sajátjaidra, ahogy szoktad)
# ---------------------------------------------------------
PROJECT_ROOT = Path().resolve().parents[1]  # Vagy ahol épp futtatod
DATA_DIR = PROJECT_ROOT / "data" / "CSV files" / "folds"
IMAGE_DIR = PROJECT_ROOT / "data" / "Databank" / "Train data" / "train_images_224_letterboxed"

BATCH_SIZE = 64 # Lehet kisebb is, hogy több step legyen az LR Finderhez
INAT_RESNET_URL = "https://tfhub.dev/google/inaturalist/resnet_v1_50/feature_vector/5"

# ---------------------------------------------------------
# ADATBETÖLTÉS (Csak a Train adatra van szükségünk ehhez)
# ---------------------------------------------------------
def process_image(file_path, class_id, venomous):
    img = tf.io.read_file(file_path)
    img = tf.io.decode_jpeg(img, channels=3, dct_method='INTEGER_FAST')
    img.set_shape([224, 224, 3])
    return img, {'species_head': class_id, 'venomous_head': venomous}

def create_train_dataset(csv_file, img_dir, batch_size):
    df = pd.read_csv(csv_file)
    file_paths = [os.path.join(str(img_dir), str(p)) for p in df['image_path']]
    ds = tf.data.Dataset.from_tensor_slices((
        file_paths, df['class_id'].values.astype('int32'), df['MIVS'].values.astype('float32')
    ))
    ds = ds.shuffle(buffer_size=1000).map(process_image, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

# ---------------------------------------------------------
# MODELL (ResNet50)
# ---------------------------------------------------------
def build_resnet_model(num_species=296):
    inputs = Input(shape=(224, 224, 3), dtype=tf.uint8)
    x = layers.Rescaling(scale=1./255, dtype='float32')(inputs)
    
    print("[+] ResNet50 iNaturalist TF Hub modul letöltése/betöltése...")
    backbone = hub.KerasLayer(INAT_RESNET_URL, trainable=False, name='resnet50_inat')
    x = backbone(x)
    
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    
    s_out = layers.Dense(num_species, activation='softmax', name='species_head')(x)
    v_out = layers.Dense(1, activation='sigmoid', name='venomous_head')(x)
    
    return models.Model(inputs=inputs, outputs=[s_out, v_out])

# ---------------------------------------------------------
# LR FINDER CALLBACK (Ez csinálja a varázslatot!)
# ---------------------------------------------------------
class LRFinder(tf.keras.callbacks.Callback):
    def __init__(self, min_lr=1e-7, max_lr=1.0, steps=200):
        super().__init__()
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.total_steps = steps
        self.step = 0
        self.lrs = []
        self.losses = []
        self.best_loss = 1e9
        
    def on_train_batch_begin(self, batch, logs=None):
        # LR exponenciális növelése
        lr = self.min_lr * (self.max_lr / self.min_lr) ** (self.step / self.total_steps)
        tf.keras.backend.set_value(self.model.optimizer.lr, lr)
        self.lrs.append(lr)
        
    def on_train_batch_end(self, batch, logs=None):
        self.step += 1
        loss = logs.get('loss')
        self.losses.append(loss)
        
        if loss < self.best_loss:
            self.best_loss = loss
            
        # Ha a loss hirtelen felrobban (túl nagy lesz), álljunk le!
        if math.isnan(loss) or loss > self.best_loss * 5:
            print(f"\n[!] A loss felrobbant (step {self.step}). LR Finder leáll.")
            self.model.stop_training = True
            
    def plot_loss(self, save_path="lr_finder_plot.png"):
        plt.figure(figsize=(10, 6))
        
        # A görbe simítása, hogy jobban olvasható legyen
        smoothed_losses = pd.Series(self.losses).rolling(window=5, min_periods=1).mean()
        
        plt.plot(self.lrs, smoothed_losses)
        plt.xscale('log')
        plt.xlabel('Learning Rate (Log skála)')
        plt.ylabel('Teljes Loss')
        plt.title('LR Finder - Keresd a legmeredekebb szakaszt!')
        plt.grid(True, which="both", ls="-", alpha=0.3)
        plt.savefig(save_path)
        print(f"[+] LR Finder grafikon elmentve: {save_path}")
        plt.show()

# ---------------------------------------------------------
# FŐ FUTTATÁS
# ---------------------------------------------------------
if __name__ == "__main__":
    train_csv = DATA_DIR / "fold_1_train.csv" # Elég egy foldon lefuttatni
    
    print("[+] Adatok betöltése...")
    train_ds = create_train_dataset(train_csv, IMAGE_DIR, BATCH_SIZE)
    
    # Hány step-ből álljon a teszt? (Több step = finomabb grafikon)
    num_steps = 300 
    
    lr_finder = LRFinder(min_lr=1e-7, max_lr=1.0, steps=num_steps)
    
    model = build_resnet_model()
    
    # Egyszerű compile, csak a loss számít most
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-7),
        loss={'species_head': 'sparse_categorical_crossentropy', 'venomous_head': 'binary_crossentropy'},
        loss_weights={'species_head': 1.0, 'venomous_head': 0.2}
    )
    
    print(f"\n=== LR Finder indítása ({num_steps} lépés) ===")
    # 1 epoch elég lehet, ha elég nagy az adathalmaz. 
    # Ha kicsi az adat, tegyél be 2-3 epochot, az lr_finder le fogja állítani a limitnél.
    model.fit(train_ds, epochs=3, callbacks=[lr_finder])
    
    # Grafikon kirajzolása
    lr_finder.plot_loss()