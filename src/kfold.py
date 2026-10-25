import pandas as pd
from sklearn.model_selection import StratifiedKFold


# 2. Stratified 5-Fold inicializálása (garantálja a class_id arányokat)
n_splits = 2
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
df_full = pd.read_csv("data/final_merged_metadata_with_new_labels.csv")

# 3. A CSV-k legenerálása
for fold, (train_idx, val_idx) in enumerate(skf.split(df_full, df_full['class_id']), 1):
    df_fold_train = df_full.iloc[train_idx]
    df_fold_val = df_full.iloc[val_idx]
    
    train_filename = f"fold_{fold}_train.csv"
    val_filename = f"fold_{fold}_val.csv"
    
    df_fold_train.to_csv(train_filename, index=False)
    df_fold_val.to_csv(val_filename, index=False)
    
    print(f"[{fold}. Fold Kész] Train: {len(df_fold_train)} | Val: {len(df_fold_val)}")