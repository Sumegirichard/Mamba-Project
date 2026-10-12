from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
import numpy as np

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def split_data():
    # Cseréld ki a pontos CSV nevedre, ha más lenne!
    df = pd.read_csv(DATA_DIR / "merged_snake_metadata.csv")
    
    # 80-20% vágás az observation_id alapján (Data Leakage védelem)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, val_idx = next(gss.split(df, groups=df['observation_id']))
    
    df_train = df.iloc[train_idx]
    df_val = df.iloc[val_idx]
    
    # Kimentjük a kész fájlokat
    df_train.to_csv(DATA_DIR / "train_split.csv", index=False)
    df_val.to_csv(DATA_DIR / "val_split.csv", index=False)
    
    print(f"Kész! Train mérete: {len(df_train)}, Val mérete: {len(df_val)}")

    # Súlyok kiszámítása a mérges kígyók ritkasága miatt
    venomous_counts = df_train['MIVS'].value_counts().sort_index().values
    weights = len(df_train) / (2 * venomous_counts)
    print(f"Tanítási súlyok - Nem mérges: {weights[0]:.2f}, Mérges: {weights[1]:.2f}")

if __name__ == "__main__":
    split_data()