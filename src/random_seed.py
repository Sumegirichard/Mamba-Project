import os
import random
import numpy as np
import tensorflow as tf

def set_global_seed(seed=42):
    # 1. Python beépített random generátorának fixálása
    random.seed(seed)
    
    # 2. Python hash seed fixálása (a szótárak sorrendje miatt)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # 3. NumPy random generátorának fixálása
    np.random.seed(seed)
    
    # 4. TensorFlow globális random seed-jének fixálása
    tf.random.set_seed(seed)
    
    # 5. Keras specifikus seed beállítása (TF 2.8+ verzióktól)
    tf.keras.utils.set_random_seed(seed)
    
    # Opcionális: Szigorú GPU determinizmus bekapcsolása
    # Figyelem: ez 10-20%-kal lelassíthatja a tanítást, mert kikapcsolja
    # a GPU bizonyos optimalizált, de nem determinisztikus műveleteit!
    # tf.config.experimental.enable_op_determinism()

# Hívd meg a függvényt a kódod legelején!
set_global_seed(42)