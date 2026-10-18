import os
import zipfile
from pathlib import Path
from PIL import Image, ImageOps
from tqdm import tqdm 

D_DRIVE_BASE = Path("D:/Letöltések")

# A Letöltések mappában lévő zip fájl pontos helye
ZIP_PATH = D_DRIVE_BASE / "train_images_medium.zip"

# Ide fogja kicsomagolni a D meghajtón
EXTRACT_DIR = D_DRIVE_BASE / "train_images_medium"

OUTPUT_DIR = D_DRIVE_BASE / "train_images_384_letterboxed"

TARGET_SIZE = (384, 384)
PAD_COLOR = (127, 127, 127)

# ---------------------------------------------------------
# FELDOLGOZÓ FÜGGVÉNY
# ---------------------------------------------------------
def process_and_save_images(input_dir, output_dir, target_size, pad_color):
    output_dir.mkdir(parents=True, exist_ok=True)
    
    valid_extensions = ('.jpg', '.jpeg', '.png')
    # A rglob('*') megkeresi a képeket akkor is, ha a zip egy almappába tette őket
    image_paths = [
        p for p in input_dir.rglob('*') 
        if p.is_file() and p.suffix.lower() in valid_extensions
    ]
    
    if not image_paths:
        print(f"Nem találtam képeket itt: {input_dir}")
        return

    print(f"Összesen {len(image_paths)} kép feldolgozása indul...")
    
    for img_path in tqdm(image_paths, desc="Képek átméretezése"):
        try:
            with Image.open(img_path) as img:
                img = img.convert("RGB")
                
                letterboxed_img = ImageOps.pad(
                    image=img, 
                    size=target_size, 
                    method=Image.Resampling.LANCZOS,
                    color=pad_color
                )
                
                save_path = output_dir / img_path.name
                letterboxed_img.save(save_path, format="JPEG", quality=100)
                
        except Exception as e:
            print(f"\nHiba történt a(z) {img_path.name} fájl feldolgozásakor: {e}")

# ---------------------------------------------------------
# FŐPROGRAM
# ---------------------------------------------------------
if __name__ == "__main__":
    # 1. Zip fájl keresése és kicsomagolása a D meghajtóra
    if not EXTRACT_DIR.exists():
        if ZIP_PATH.exists():
            print(f"Zip fájl megtalálva! Kicsomagolás indul: {ZIP_PATH} -> {EXTRACT_DIR}")
            EXTRACT_DIR.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
                zip_ref.extractall(EXTRACT_DIR)
            print("Kicsomagolás kész!\n")
        else:
            print(f"HIBA: Nem található a zip fájl a megadott helyen: {ZIP_PATH}")
            exit()
            
    # 2. Képek átméretezése és mentése a D meghajtóra
    process_and_save_images(EXTRACT_DIR, OUTPUT_DIR, TARGET_SIZE, PAD_COLOR)
    print(f"\nA feldolgozás befejeződött! A feltöltendő képek itt találhatók: {OUTPUT_DIR}")