import pandas as pd
from sklearn.model_selection import StratifiedKFold

# 1. Beolvasás és az eddigi fix split egyesítése a teljes adathalmazzá
df_train = pd.read_csv("train_split_fixed.csv")
df_val = pd.read_csv("val_split_fixed.csv")
df_full = pd.concat([df_train, df_val], ignore_index=True)

print(f"Teljes adathalmaz mérete: {len(df_full)} kép")

# 2. Stratified 5-Fold inicializálása (garantálja a class_id arányokat)
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

# 3. A CSV-k legenerálása
for fold, (train_idx, val_idx) in enumerate(skf.split(df_full, df_full['class_id']), 1):
    df_fold_train = df_full.iloc[train_idx]
    df_fold_val = df_full.iloc[val_idx]
    
    train_filename = f"fold_{fold}_train.csv"
    val_filename = f"fold_{fold}_val.csv"
    
    df_fold_train.to_csv(train_filename, index=False)
    df_fold_val.to_csv(val_filename, index=False)
    
    print(f"[{fold}. Fold Kész] Train: {len(df_fold_train)} | Val: {len(df_fold_val)}")