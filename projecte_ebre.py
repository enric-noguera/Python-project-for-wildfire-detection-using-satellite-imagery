# -*- coding: utf-8 -*-
"""
Created on Thu Jul 10 11:21:40 2025

"""
# -*- coding: utf-8 -*-
# Projecte PIVA - Torre de l'Espanyol
# NDVI, NBR, dNBR, severitat, superfície cremada, i model Random Forest

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.colors as colors
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# ================================================
# 1. Funcions per calcular índexs espectrals
# ================================================

def calcular_ndvi(nir, red):
    return (nir - red) / (nir + red + 1e-10)

def calcular_nbr(nir, swir2):
    return (nir - swir2) / (nir + swir2 + 1e-10)

def calcular_savi(nir, red, L=0.5):
    return ((nir - red) / (nir + red + L)) * (1 + L)

def calcular_evi(nir, red, blue):
    return 2.5 * (nir - red) / (nir + 6 * red - 7.5 * blue + 1)

def calcular_gci(nir, green):
    return (nir / (green + 1e-10)) - 1

# ================================================
# 2. Funció per llegir i retallar les bandes
# ================================================

def llegir_bandes(rutas, factor=10):
    bandes = []
    for ruta in rutas:
        # Llegir la banda amb matplotlib.image
        banda = mpimg.imread(ruta)
        # Si la imatge té múltiples canals, agafar només el primer
        if len(banda.shape) > 2:
            banda = banda[:, :, 0]
        # Convertir a float per evitar problemes amb divisions
        banda = banda.astype(float)
        # Aplicar factor de mostreig si és necessari
        if factor > 1:
            banda = banda[::factor, ::factor]
        bandes.append(banda)
    return bandes

# ================================================
# 3. Carregar bandes pre i post incendi
# ================================================

rutas_previ = [
    "LC08_L1TP_198031_20180624_20200831_02_T1_B2.TIF",  # Blue
    "LC08_L1TP_198031_20180624_20200831_02_T1_B3.TIF",  # Green
    "LC08_L1TP_198031_20180624_20200831_02_T1_B4.TIF",  # Red
    "LC08_L1TP_198031_20180624_20200831_02_T1_B5.TIF",  # NIR
    "LC08_L1TP_198031_20180624_20200831_02_T1_B7.TIF"   # SWIR2
]

# Fitxers després de l'incendi (post-foc)
rutas_post = [
    "LC08_L1TP_198031_20190814_20200827_02_T1_B2.TIF",  # Blue
    "LC08_L1TP_198031_20190814_20200827_02_T1_B3.TIF",  # Green
    "LC08_L1TP_198031_20190814_20200827_02_T1_B4.TIF",  # Red
    "LC08_L1TP_198031_20190814_20200827_02_T1_B5.TIF",  # NIR
    "LC08_L1TP_198031_20190814_20200827_02_T1_B7.TIF"   # SWIR2
]

# Llegir les bandes
bandes_previ = llegir_bandes(rutas_previ, factor=10)
bandes_post = llegir_bandes(rutas_post, factor=10)

# Retallar a la mida comuna (720x780)
B2_pre, B3_pre, B4_pre, B5_pre, B7_pre = [r[0:720, 0:780] for r in bandes_previ]
B2_post, B3_post, B4_post, B5_post, B7_post = [r[0:720, 0:780] for r in bandes_post]

# ================================================
# 4. Càlcul d'índexs espectrals i dNBR
# ================================================

rgb_previ = np.dstack((B4_pre, B3_pre, B2_pre))
rgb_post = np.dstack((B4_post, B3_post, B2_post))
rgb_previ_norm = rgb_previ / np.percentile(rgb_previ, 98)
rgb_post_norm = rgb_post / np.percentile(rgb_post, 98)

ndvi_pre = calcular_ndvi(B5_pre, B4_pre)
ndvi_post = calcular_ndvi(B5_post, B4_post)
diferencia_ndvi = ndvi_pre - ndvi_post

ndvi_2018_retallat = ndvi_pre[500:650, 100:250]
ndvi_2019_retallat = ndvi_post[500:650, 100:250]
diferencia_ndvi_retallat = diferencia_ndvi[500:650, 100:250]

nbr_pre = calcular_nbr(B5_pre, B7_pre)
nbr_post = calcular_nbr(B5_post, B7_post)
dnbr = nbr_pre - nbr_post

nbr_2018_retallat = nbr_pre[500:650, 100:250]
nbr_2019_retallat = nbr_post[500:650, 100:250]
diferencia_nbr_retallat = dnbr[500:650, 100:250]

savi_pre = calcular_savi(B5_pre, B4_pre)
gci_pre = calcular_gci(B5_pre, B3_pre)
evi_pre = calcular_evi(B5_pre, B4_pre, B2_pre)

# ================================================
# 5. Classificació de severitat segons dNBR
# ================================================

def classificar_severitat(dnbr):
    classificacio = np.zeros_like(dnbr)
    classificacio[(dnbr >= -0.5) & (dnbr < -0.251)] = 1
    classificacio[(dnbr >= -0.25) & (dnbr < -0.101)] = 2
    classificacio[(dnbr >= -0.1) & (dnbr <= 0.099)] = 3
    classificacio[(dnbr >= 0.1) & (dnbr < 0.269)] = 4
    classificacio[(dnbr >= 0.27) & (dnbr < 0.439)] = 5
    classificacio[(dnbr >= 0.44) & (dnbr < 0.659)] = 6
    classificacio[(dnbr >= 0.66) & (dnbr <= 1.3)] = 7
    return classificacio

severitat = classificar_severitat(dnbr)

# ================================================
# 6. Superfície cremada
# ================================================

mask_cremat = dnbr > 0.1
pixels_cremats = np.sum(mask_cremat)
area_km2 = pixels_cremats * 0.0009
print(f"Superfície cremada estimada: {area_km2:.2f} km^2")

# ================================================
# 7. Model Random Forest per predicció de severitat
# ================================================

X = np.stack([
    ndvi_pre.flatten(),
    savi_pre.flatten(),
    gci_pre.flatten(),
    evi_pre.flatten()
], axis=1)

y = dnbr.flatten()

mask_valid = np.all(np.isfinite(X), axis=1) & np.isfinite(y)
X_valid = X[mask_valid]
y_valid = y[mask_valid]

X_train, X_test, y_train, y_test = train_test_split(X_valid, y_valid, test_size=0.3, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=0)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(f"R^2 del model Random Forest: {r2_score(y_test, y_pred):.2f}")

# Fem una predicció sobre tota la imatge
X_pred_full = np.stack([
    ndvi_pre.flatten(),
    savi_pre.flatten(),
    gci_pre.flatten(),
    evi_pre.flatten()
], axis=1)

mask_pred = np.all(np.isfinite(X_pred_full), axis=1)
X_pred_clean = X_pred_full[mask_pred]
predicted_dnbr_flat = np.zeros(X_pred_full.shape[0])
predicted_dnbr_flat[mask_pred] = model.predict(X_pred_clean)

# Reconstruïm la imatge predita
predicted_dnbr = predicted_dnbr_flat.reshape(ndvi_pre.shape)

# ================================================
# 8. Visualització completa
# ================================================

plt.figure(figsize=(12,12))
plt.subplot(2,2,1)
plt.imshow(B4_post, cmap='gray')
plt.title("Banda 4 (Red)")
plt.axis('off')

plt.subplot(2,2,2)
plt.imshow(B5_post, cmap='gray')
plt.title("Banda 5 (NIR)")
plt.axis('off')

plt.subplot(2,2,3)
plt.imshow(B7_post, cmap='gray')
plt.title("Banda 7 (SWIR2)")
plt.axis('off')

plt.subplot(2,2,4)
plt.imshow(B2_post, cmap='gray')
plt.title("Banda 2 (Blue)")
plt.axis('off')

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(np.clip(rgb_previ_norm, 0, 1))
plt.title("RGB pre-foc")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(np.clip(rgb_post_norm, 0, 1))
plt.title("RGB post-foc")
plt.axis('off')

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(ndvi_2018_retallat, cmap="jet", vmin=np.min(ndvi_2018_retallat), vmax=np.max(ndvi_2018_retallat))
plt.colorbar()
plt.title("NDVI 2018")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(ndvi_2019_retallat, cmap="jet", vmin=np.min(ndvi_2019_retallat), vmax=np.max(ndvi_2019_retallat))
plt.colorbar()
plt.title("NDVI 2019")
plt.axis('off')

plt.figure(figsize=(6, 6))
plt.imshow(diferencia_ndvi_retallat, cmap="RdYlGn", vmin=np.min(diferencia_ndvi_retallat), vmax=np.max(diferencia_ndvi_retallat))
plt.title("Diferència NDVI")
plt.colorbar()
plt.axis('off')

plt.figure(figsize=(10,5))
plt.subplot(1, 2, 1)
plt.imshow(nbr_2018_retallat, cmap="RdYlGn", vmin=np.min(nbr_2019_retallat), vmax=np.max(nbr_2019_retallat))
plt.colorbar()
plt.title("NBR 2018")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(nbr_2019_retallat, cmap="RdYlGn", vmin=np.min(nbr_2019_retallat), vmax=np.max(nbr_2019_retallat))
plt.colorbar()
plt.title("NBR 2019")
plt.axis('off')

plt.figure(figsize=(6, 5))
plt.imshow(diferencia_nbr_retallat, cmap="RdYlGn", vmin=np.min(diferencia_nbr_retallat), vmax=np.max(diferencia_nbr_retallat))
plt.title("Diferència NBR")
plt.colorbar()
plt.axis('off')

cmap = colors.ListedColormap(['gray', 'green', 'lightgreen', 'white', 'yellow', 'orange', 'red'])
bounds = [0, 1, 2, 3, 4, 5, 6, 7]
norm = colors.BoundaryNorm(bounds, cmap.N)

plt.figure(figsize=(8, 6))
plt.imshow(severitat[500:650, 100:250], cmap=cmap, norm=norm)
plt.colorbar(ticks=bounds, label='Severitat')
plt.title("Classificació de severitat d'incendi (dNBR)")
plt.axis('off')
plt.tight_layout()
plt.show()

# Mapa de dNBR predit
plt.figure(figsize=(8, 6))
plt.imshow(predicted_dnbr, cmap="RdYlGn", vmin=-1, vmax=1)
plt.title("Predicció de dNBR amb Random Forest")
plt.colorbar(label='dNBR predit')
plt.axis('off')
plt.tight_layout()
plt.show()