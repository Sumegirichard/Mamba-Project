import tensorflow as tf
import numpy as np
import os
from pathlib import Path

# Felesleges TF logok elrejtése
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "snake_multitask_model.h5"


def preprocess_single_image(image_path):
    # 1. Kép betöltése és előkészítése (pontosan úgy, mint a tanításnál!)
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [224, 224])
    img = img / 255.0  # Normalizálás 0-1 közé
    
    # KULCSLÉPÉS: A modell "batch"-eket (paklikat) vár. 
    # Mivel ez csak 1 db kép (224, 224, 3), csinálunk belőle egy 1 lapos paklit: (1, 224, 224, 3)
    img = tf.expand_dims(img, axis=0) 
    return img

def test_prediction(image_path):
    print(f"[{image_path}] feldolgozása...\n")
    
    # 2. Modell betöltése
    # compile=False: Mivel csak jósolni akarunk, nem kell neki a Loss függvény!
    model = tf.keras.models.load_model(str(MODEL_PATH), compile=False)
    
    # 3. Kép átalakítása tenzorrá
    img_tensor = preprocess_single_image(image_path)
    
    # 4. Jóslás (Predict) végrehajtása
    predictions = model.predict(img_tensor)
    
    # A modellünk 2 kimenetet ad (Faj, Mérgezőség), így a 'predictions' egy 2 elemű lista
    species_preds = predictions[0][0]     # A 296 fajhoz tartozó valószínűségek
    venomous_pred = predictions[1][0][0]  # Egyetlen szám 0.0 és 1.0 között
    
    # 5. Eredmények kiértékelése
    # Megkeressük a legnagyobb valószínűségű faj indexét (Class ID)
    predicted_class_id = np.argmax(species_preds) 
    species_confidence = species_preds[predicted_class_id] * 100
    
    # Mérgezőség értelmezése (50% felett mérgesnek tekintjük)
    is_venomous = "MÉRGES" if venomous_pred > 0.5 else "NEM mérges"
    # Kiszámoljuk, mennyire biztos benne
    ven_conf = venomous_pred * 100 if venomous_pred > 0.5 else (1 - venomous_pred) * 100

    # 6. Eredmény kiíratása
    print("\n" + "="*40)
    print("🐍 A MODELL ÍTÉLETE:")
    print("="*40)
    print(f"Felismerve mint (Class ID) : {predicted_class_id}")
    print(f"Faj biztossági szint       : {species_confidence:.2f}%")
    print("-" * 40)
    print(f"Mérgezőség                 : {is_venomous}")
    print(f"Mérgezőség biztossága      : {ven_conf:.2f}%")
    print("="*40)

if __name__ == "__main__":
    # Írd ide egy olyan kép nevét, amit be akarsz adni a gépnek!
    # (Lehet a 'snake_images_small' mappából egy fájl, vagy egy internetről letöltött .jpg is)
    TESZT_KEP = str(PROJECT_ROOT / "data" / "train_images_small" / "train_images_small" / "3270482.jpeg") #Eunectes murinus; 0 -nem mérgező
    
    try:
        test_prediction(TESZT_KEP)
    except FileNotFoundError:
        print(f"Hiba: Nem találom a '{TESZT_KEP}' fájlt. Írd át az elérési utat!")