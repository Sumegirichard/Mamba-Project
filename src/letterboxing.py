import os
from pathlib import Path
from PIL import Image, ImageOps
from tqdm import tqdm  # Csomag telepítése, ha nincs: pip install tqdm

# ---------------------------------------------------------
# BEÁLLÍTÁSOK
# ---------------------------------------------------------
# Az elérési utakat a saját struktúrádhoz igazítsd
try:
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
except NameError:
    # Jupyter Notebook kompatibilitás miatt
    PROJECT_ROOT = Path().resolve().parents[1] 

DATA_DIR = PROJECT_ROOT / "data"
# A mappák neveit cseréld ki a megfelelőre!
INPUT_DIR = DATA_DIR / "train_images_small" / "train_images_small"
OUTPUT_DIR = DATA_DIR / "train_images_224_letterboxed"

TARGET_SIZE = (224, 224)
PAD_COLOR = (127, 127, 127)  # Szürke padding a konvolúciós hálózatokhoz

# ---------------------------------------------------------
# FELDOLGOZÓ FÜGGVÉNY
# ---------------------------------------------------------
def process_and_save_images(input_dir, output_dir, target_size, pad_color):
    # Létrehozzuk a kimeneti mappát, ha nem létezik
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Kigyűjtjük az összes .jpg, .jpeg, .png fájlt
    valid_extensions = ('.jpg', '.jpeg', '.png')
    image_paths = [
        p for p in input_dir.rglob('*') 
        if p.is_file() and p.suffix.lower() in valid_extensions
    ]
    
    print(f"Összesen {len(image_paths)} kép feldolgozása indul...")
    
    # tqdm a folyamatjelző sávhoz
    for img_path in tqdm(image_paths, desc="Képek átméretezése"):
        try:
            with Image.open(img_path) as img:
                # Biztosítjuk, hogy a kép RGB formátumú legyen
                img = img.convert("RGB")
                
                # Letterboxing alkalmazása
                # ImageOps.pad arányosan skáláz, és kipótolja a széleket
                letterboxed_img = ImageOps.pad(
                    image=img, 
                    size=target_size, 
                    method=Image.Resampling.LANCZOS, # Magas minőségű skálázás
                    color=pad_color
                )
                
                # Mentési útvonal generálása (megtartja az eredeti fájlnevet)
                # Ha almappák is vannak, ezt érdemes finomítani
                save_path = output_dir / img_path.name
                
                # Kép elmentése magas minőségben
                letterboxed_img.save(save_path, format="JPEG", quality=95)
                
        except Exception as e:
            print(f"\nHiba történt a(z) {img_path.name} fájl feldolgozásakor: {e}")

# ---------------------------------------------------------
# FŐPROGRAM
# ---------------------------------------------------------
if __name__ == "__main__":
    if not INPUT_DIR.exists():
        print(f"A megadott bemeneti mappa nem létezik: {INPUT_DIR}")
    else:
        process_and_save_images(INPUT_DIR, OUTPUT_DIR, TARGET_SIZE, PAD_COLOR)
        print(f"\nA feldolgozás befejeződött! A képek itt találhatók: {OUTPUT_DIR}")