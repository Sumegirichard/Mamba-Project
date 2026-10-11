import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import matplotlib.image as mpimg


df = pd.read_csv("merged_snake_metadata.csv")
print(f"Sorok száma: {df.shape[0]}, Oszlopok száma: {df.shape[1]}")

print(df.info())
print(df.head())

duplicates = df.duplicated().sum()
print(f"Teljes soros duplikátumok: {duplicates}")

osszes_sor = len(df)
print(f"Összes sor száma: {osszes_sor}")


missing = df.isnull().sum()
missing_pct = (missing / len(df)) * 100
missing_df = pd.DataFrame({"Hiányzó darabszám": missing, "Hiányzó arány (%)": missing_pct})
print(missing_df[missing_df["Hiányzó darabszám"] > 0])

df = pd.read_csv("merged_snake_metadata.csv")

# 2. Összes sor és oszlop száma
print(f"Összes sor (kép) száma: {len(df)}")
print(f"Oszlopok: {df.columns.tolist()}")

# 3. Különböző osztályok száma azonosító és tudományos név szerint
osztalyok_id_szerint = df["class_id"].nunique()
osztalyok_nev_szerint = df["binomial_name"].nunique()

print(f"Különböző osztályok száma (class_id): {osztalyok_id_szerint}")
print(f"Különböző fajok száma (binomial_name): {osztalyok_nev_szerint}")

# 4. A leggyakoribb és legritkább 5 osztály megtekintése
print("\n--- Top 5 leggyakoribb faj ---")
print(df["binomial_name"].value_counts().head(5))

print("\n--- 5 legritkább faj ---")
print(df["binomial_name"].value_counts().tail(5))

egyedi_kepek_szama = df["image_path"].nunique()

# 1. Adatok beolvasása
df = pd.read_csv("merged_snake_metadata.csv")
faj_eloszlas = df["binomial_name"].value_counts()

# 2. Ábrák létrehozása egymás mellett
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

# --- Bal oldali ábra: Top 15 leggyakoribb faj ---
top15 = faj_eloszlas.head(15).sort_values(ascending=True)
bars = ax1.barh(top15.index, top15.values, color="#2b5c8f", edgecolor="black", alpha=0.85)

# Értékek ráírása az oszlopok végére
for bar in bars:
    w = bar.get_width()
    ax1.text(w + 15, bar.get_y() + bar.get_height() / 2, f"{int(w)}",
             ha="left", va="center", fontsize=9)

ax1.set_title("Top 15 leggyakoribb kígyófaj", fontsize=13, fontweight="bold")
ax1.set_xlabel("Képek száma (db)", fontsize=11)
ax1.grid(axis="x", linestyle="--", alpha=0.5)

# --- Jobb oldali ábra: Teljes eloszlás (hisztogram) ---
ax2.hist(faj_eloszlas.values, bins=35, color="#d95f02", edgecolor="black", alpha=0.8)
ax2.axvline(faj_eloszlas.mean(), color="black", linestyle="--", linewidth=1.5,
            label=f"Átlag: {faj_eloszlas.mean():.1f}")
ax2.axvline(faj_eloszlas.median(), color="blue", linestyle=":", linewidth=1.5,
            label=f"Medián: {faj_eloszlas.median():.1f}")

ax2.set_title("Fajonkénti képszám eloszlása (296 faj)", fontsize=13, fontweight="bold")
ax2.set_xlabel("Képek száma egy adott fajhoz", fontsize=11)
ax2.set_ylabel("Fajok darabszáma", fontsize=11)
ax2.grid(axis="y", linestyle="--", alpha=0.5)
ax2.legend()

plt.tight_layout()
plt.show()


import zipfile

ZIP_PATH = "train_images_small.zip"
TARGET_SIZE = (224, 224)
MAX_IMAGES = 32  # Ennyit olvasunk be mintának

kepek = []

with zipfile.ZipFile(ZIP_PATH, "r") as z:
    file_list = [f for f in z.namelist() if f.lower().endswith(('.jpg', '.jpeg', '.png')) and not f.startswith('__MACOSX')]
    
    print(f"Összes talált kép a zipben: {len(file_list)} db")
    
    for f in file_list[:MAX_IMAGES]:
        print(f)
        raw_bytes = z.read(f)
        # Dekódolás 3 csatornára (RGB)
        img = tf.io.decode_image(raw_bytes, channels=3, expand_animations=False)
        img = tf.image.resize(img, TARGET_SIZE)
        
        # --- NORMALIZÁLÁS ---
        img = img / 255.0
        
        kepek.append(img)

# Összefűzés egyetlen 4D tenzorrá: (darabszám, magasság, szélesség, 3)
kepek_tenzor = tf.stack(kepek)
print("Beolvasott tenzor alakja:", kepek_tenzor.shape)

# Így már 0 és 1 közötti értékeket fogsz látni a kiíratásnál
print(kepek_tenzor[0])


import tensorflow as tf
from tensorflow.keras import layers, models, Input

def build_custom_snake_model(num_species=296):
    # 1. Bemeneti réteg (Ezt a formátumot készítettétek elő: 224x224 pixel, 3 színcsatorna)
    inputs = Input(shape=(224, 224, 3), name='image_input')
    
    # ==========================================
    # KÖZÖS TÖRZS (Feature Extraction)
    # ==========================================
    # 1. Konvolúciós blokk (Alap formák, élek felismerése)
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = layers.MaxPooling2D((2, 2))(x)
    
    # 2. Konvolúciós blokk (Összetettebb minták, pl. pikkelyek)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    
    # 3. Konvolúciós blokk (Testrészek, fejforma felismerése)
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    
    # 4. Konvolúciós blokk
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    
    # A 4D tenzort kilapítjuk 1D vektorrá a neurális háló fejei számára
    x = layers.GlobalAveragePooling2D()(x) 
    
    # Egy sűrű (Dense) réteg, hogy a kinyert jellemzőket kombinálja
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.5)(x) # 50% "felejtés" a túltanulás (overfitting) ellen

    # ==========================================
    # KETTÉVÁLÁS: A két "Fej" (Output Heads)
    # ==========================================
    
    # A) Fej: Faj felismerése (Többosztályos klasszifikáció)
    # Softmax aktiváció: 296 valószínűséget ad, amiknek az összege 1 (100%)
    species_output = layers.Dense(
        num_species, 
        activation='softmax', 
        name='species_head'
    )(x)
    
    # B) Fej: Mérgezőség (Bináris klasszifikáció)
    # Sigmoid aktiváció: 1 db számot ad 0 és 1 között (pl. 0.85 = 85%, hogy mérges)
    venomous_output = layers.Dense(
        1, 
        activation='sigmoid', 
        name='venomous_head'
    )(x)
    
    # A teljes modell összekapcsolása
    model = models.Model(
        inputs=inputs, 
        outputs=[species_output, venomous_output],
        name="Custom_Snake_CNN"
    )
    
    return model

# Modell inicializálása és struktúra kiíratása
model = build_custom_snake_model(num_species=296)
model.summary()
