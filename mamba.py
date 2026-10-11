
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import matplotlib.image as mpimg

# Beolvasás (egy NumPy array lesz: magasság x szélesség x színcsatornák)
#img = mpimg.imread("ide ird ricsi .jpg")  # Cseréld ki a fájlnevet a saját képedre

# Megjelenítés
#plt.figure(figsize=(6, 6))
#plt.imshow(img)
#plt.axis("off")
#plt.show()

df = pd.read_csv("merged_snake_metadata.csv")
print(f"Sorok száma: {df.shape[0]}, Oszlopok száma: {df.shape[1]}")

print(df.info())
print(df.head())

duplicates = df.duplicated().sum()
print(f"Teljes soros duplikátumok: {duplicates}")






