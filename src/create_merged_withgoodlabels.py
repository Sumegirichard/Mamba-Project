import pandas as pd
import os

def create_mapped_dataset(metadata_path, mapping_path, output_path):
    print(f"Betöltés: {metadata_path} és {mapping_path} ...")
    
    # Fájlok beolvasása
    metadata_df = pd.read_csv(metadata_path)
    mapping_df = pd.read_csv(mapping_path)
    
    # Merge (Összekapcsolás) a megfelelő oszlopok alapján
    merged_df = metadata_df.merge(
        mapping_df, 
        left_on='class_id', 
        right_on='original_class_id', 
        how='left'
    )
    
    # Régi class_id felülírása az új new_model_id-vel
    merged_df['class_id'] = merged_df['new_model_id'].astype('Int64')
    
    # Feleslegessé vált oszlopok törlése
    merged_df = merged_df.drop(columns=['original_class_id', 'new_model_id'])
    
    # Eredmény mentése
    merged_df.to_csv(output_path, index=False)
    print(f"Kész! Az új fájl elmentve ide: {output_path}")

if __name__ == "__main__":
    META_FILE = 'data/merged_snake_metadata.csv'
    MAPPING_FILE = 'data/class_id_mapping.csv'
    OUTPUT_FILE = 'data/final_merged_metadata_with_new_labels.csv'
    
    create_mapped_dataset(META_FILE, MAPPING_FILE, OUTPUT_FILE)