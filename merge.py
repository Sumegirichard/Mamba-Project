import pandas as pd

def merge_metadata(train_csv_path, venomous_csv_path, output_csv_path):
    # CSV fájlok beolvasása
    df_train = pd.read_csv(train_csv_path)
    df_venomous = pd.read_csv(venomous_csv_path)

    # Összekapcsolás a class_id alapján
    df_merged = pd.merge(df_train, df_venomous, on='class_id', how='left')

    # Felesleges index oszlopok eltávolítása, ha léteznek
    columns_to_drop = [col for col in df_merged.columns if 'Unnamed: 0' in col]
    if columns_to_drop:
        df_merged = df_merged.drop(columns=columns_to_drop)

    # Mentés az új CSV fájlba
    df_merged.to_csv(output_csv_path, index=False)
    print(f"Az adatok sikeresen összefűzve és mentve ide: {output_csv_path}")

if __name__ == "__main__":
    train_path = 'train_images_metadata.csv'
    venomous_path = 'venomous_status_metadata.csv'
    output_path = 'merged_snake_metadata.csv'
    
    merge_metadata(train_path, venomous_path, output_path)