import pandas as pd
from sklearn.preprocessing import LabelEncoder

def fix_class_ids():
    # 1. Betöltjük az eddig szétválasztott fájlokat
    # Cseréld ki a neveket, ha nem train_split.csv és val_split.csv a nevük!
    train_df = pd.read_csv("train_split.csv")
    val_df = pd.read_csv("val_split.csv")
    
    # 2. Összevonjuk őket ideiglenesen, hogy a LabelEncoder az összes fajt lássa
    full_df = pd.concat([train_df, val_df], ignore_index=True)
    
    # 3. Létrehozzuk az átkódolót
    le = LabelEncoder()
    le.fit(full_df['class_id'])
    
    # 4. Átírjuk a régi (nagy) ID-kat az új (0-295 közötti) ID-kra
    train_df['class_id'] = le.transform(train_df['class_id'])
    val_df['class_id'] = le.transform(val_df['class_id'])
    
    # 5. Elmentjük a felülírt fájlokat
    train_df.to_csv("train_split_fixed.csv", index=False)
    val_df.to_csv("val_split_fixed.csv", index=False)
    
    # Lementjük a szótárat is, hogy a predikciónál vissza tudjuk kódolni a számot a faj nevére!
    # Létrehozunk egy mapping DataFrame-et
    mapping_df = pd.DataFrame({
        'original_class_id': le.classes_,
        'new_model_id': range(len(le.classes_))
    })
    mapping_df.to_csv("class_id_mapping.csv", index=False)
    
    print("Sikeresen átkódolva! A fajok új ID-jai 0-tól 295-ig tartanak.")
    print("Mentett fájlok:")
    print("- train_split_fixed.csv")
    print("- val_split_fixed.csv")
    print("- class_id_mapping.csv (Ezt őrizzétek meg a jósláshoz!)")

if __name__ == "__main__":
    fix_class_ids()