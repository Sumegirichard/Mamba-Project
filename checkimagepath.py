import pandas as pd

def check_image_duplicates(csv_file):
    print(f"Adatok betöltése a '{csv_file}' fájlból...")
    df = pd.read_csv(csv_file)
    
    total_rows = len(df)
    unique_images = df['image_path'].nunique()
    num_duplicates = total_rows - unique_images
    
    print("-" * 40)
    print(f"Összes sor a fájlban: {total_rows}")
    print(f"Egyedi kép útvonalak (image_path) száma: {unique_images}")
    print(f"Duplikált image_path-ok száma: {num_duplicates}")
    print("-" * 40)
    
    if num_duplicates > 0:
        print("\nFigyelem! Duplikációkat találtam. Íme néhány példa:")
        # Kigyűjtjük az összes olyan sort, ahol az image_path többször szerepel
        duplicates = df[df.duplicated(subset=['image_path'], keep=False)]
        print(duplicates.sort_values(by='image_path')[['observation_id', 'image_path', 'class_id']].head(10))
    else:
        print("\nSzuper! Nincs egyetlen duplikált image_path sem az adatbázisban. Minden kép egyedi.")

if __name__ == "__main__":
    check_image_duplicates("merged_snake_metadata.csv")