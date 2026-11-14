from pathlib import Path
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

# Csoportos Stratified 5-Fold: egy megfigyelés (observation_id) összes képe mindig ugyanabba a halmazba kerül.
# A sima StratifiedKFold a megfigyelés képeit szétszórta a train és a val között (csoportos szivárgás),
# ezért a validációs metrikák optimisták voltak. A régi data/folds/ érintetlen marad.
DATA_DIR = Path(__file__).resolve().parents[1] / "data"
OUT_DIR = DATA_DIR / "folds_grouped"
OUT_DIR.mkdir(exist_ok=True)

n_splits = 5
df_full = pd.read_csv(DATA_DIR / "final_merged_metadata_with_new_labels.csv")
sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)

for fold, (train_idx, val_idx) in enumerate(sgkf.split(df_full, df_full['class_id'], groups=df_full['observation_id']), 1):
    df_fold_train = df_full.iloc[train_idx]
    df_fold_val = df_full.iloc[val_idx]

    # Ellenőrzés: nincs közös megfigyelés és kép a két halmaz között
    assert not set(df_fold_train['observation_id']) & set(df_fold_val['observation_id'])
    assert not set(df_fold_train['image_path']) & set(df_fold_val['image_path'])

    df_fold_train.to_csv(OUT_DIR / f"fold_{fold}_train.csv", index=False)
    df_fold_val.to_csv(OUT_DIR / f"fold_{fold}_val.csv", index=False)

    missing = df_full['class_id'].nunique() - df_fold_val['class_id'].nunique()
    print(f"[{fold}. Fold Kész] Train: {len(df_fold_train)} | Val: {len(df_fold_val)} | "
          f"Val fajok: {df_fold_val['class_id'].nunique()} (hiányzik: {missing}) | "
          f"Mérgező arány val: {df_fold_val['MIVS'].mean():.3f}")
