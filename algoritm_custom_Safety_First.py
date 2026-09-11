import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error

# CONFIGURARE DIRECTOARE
DIR_PROCESATE = '02_Date_Procesate'
DIR_MODELE = '03_Modele_Salvate'
DIR_GRAFICE = '04_Grafice_Rezultate'

# 1. ÎNCĂRCAREA DATELOR DE VALIDARE (pentru experimentare) SI CEL MAI BUN MODEL (Random Forest)
X_val = joblib.load(os.path.join(DIR_PROCESATE, 'X_val.data'))
y_val = joblib.load(os.path.join(DIR_PROCESATE, 'y_val.data'))

# ÎNCĂRCAREA DATELELOR DE TEST
x_test = joblib.load(os.path.join(DIR_PROCESATE, 'X_test.data'))
y_test = joblib.load(os.path.join(DIR_PROCESATE, 'y_test.data'))

model_rf = joblib.load(os.path.join(DIR_MODELE, 'model_RF.joblib'))

# 2. DEFINIREA CLASEI CUSTOM
class SafetyFirstPredictor:
    def __init__(self, model_baza, safety_factor=3.0):
        self.model = model_baza
        self.safety_factor = safety_factor
        self.mae_istoric = 0.0  # Eroarea medie învățată

    def fit_calibration(self, x_val, y_val):
        # Calibrăm algoritmul: vedem cât de mult greșește AI-ul de obicei
        preds = self.model.predict(x_val)
        self.mae_istoric = mean_absolute_error(y_val, preds)
        print(f"[CALIBRARE] Eroarea medie (MAE) a modelului de bază: {self.mae_istoric:.5f} µT")
        print(f"[CALIBRARE] Marja de siguranță aplicată va fi: {self.safety_factor * self.mae_istoric:.5f} µT")

    def predict_safe(self, x):
        # 1. Se face predicția RF standard
        pred_base = self.model.predict(x)

        # 2. Se aplică marja de siguranță
        # Formula: predicție + (factor * eroare_medie)
        bias = self.safety_factor * self.mae_istoric
        pred_safe = pred_base + bias

        return pred_base, pred_safe


# 3. RULAREA ALGORITMULUI
# Inițializarea algoritmului custom
custom_algo = SafetyFirstPredictor(model_rf, safety_factor=3.0)


# ==========================================================
# Implemenare pe datele de validare (pentru calibrarea safety_factor)
# ==========================================================

# Calibrare pe datele de validare
custom_algo.fit_calibration(X_val, y_val)

# Predicție pe datele de validare
y_pred_ai, y_pred_safe = custom_algo.predict_safe(X_val)

# 4. ANALIZA DE SIGURANȚĂ
# De câte ori RF-ul simplu a subestimat radiația (valoare reală > predicție)
subestimari_ai = np.sum(y_pred_ai < y_val)
# De câte ori algoritmul custom a subestimat
subestimari_safe = np.sum(y_pred_safe < y_val)

print(f"\n--- REZULTATE PE VALIDARE ---")
print(f"Total măsurători: {len(y_val)}")
print(f"Subestimări RF simplu: {subestimari_ai}")
print(f"Subestimări Safety-First: {subestimari_safe}")

if subestimari_ai > 0:
    reducere = 100 * (subestimari_ai - subestimari_safe) / subestimari_ai
    print(f"Riscul a fost redus cu {reducere:.1f}%")

# 5. GRAFIC COMPARATIV (cu zona de siguranță)
plt.figure(figsize=(12, 7))

# Selectare subset de puncte (ex: primele 50 sau 100) pentru a se vedea clar pe grafic
subset = 60
indices = range(subset)

plt.plot(indices, y_val[:subset], 'k-o', label='Valoare Reală', linewidth=2, markersize=5)
plt.plot(indices, y_pred_ai[:subset], 'r--', label='Predicție RF Standard', alpha=0.7)
plt.plot(indices, y_pred_safe[:subset], 'g-^', label='Predicție Safety-First', linewidth=2)

# Colorare zonă de siguranță
plt.fill_between(indices, y_pred_ai[:subset], y_pred_safe[:subset], color='green', alpha=0.1,
                 label='Marjă de Siguranță')

plt.title('Algoritm Hibrid "Safety-First" vs RF Standard')
plt.xlabel('Eșantioane (exemple din setul de validare)')
plt.ylabel('Inducție Magnetică (µT)')
plt.legend()
plt.grid(True)
plt.tight_layout()

# Salvare și afișare
grafic_safe_path = os.path.join(DIR_GRAFICE, 'Grafic_Safety_First_Validare.png')
plt.savefig(grafic_safe_path)
print(f"[INFO] Graficul Safety-First a fost salvat în: {grafic_safe_path}")

plt.show(block=True)


# # ==========================================================
# # Implemenatere pe datele de testare
# # ==========================================================
#
# Calibarare pe datele de test
custom_algo.fit_calibration(x_test, y_test)

# Predicții pe datele de test
y_pred_ai, y_pred_safe = custom_algo.predict_safe(x_test)

# 4. ANALIZA DE SIGURANȚĂ
subestimari_ai = np.sum(y_pred_ai < y_test)
subestimari_safe = np.sum(y_pred_safe < y_test)

print(f"\n--- REZULTATE DE SIGURANȚĂ---")
print(f"Total măsurători testate: {len(y_test)}")
print(f"Cazuri unde RF-ul simplu a subestimat riscul: {subestimari_ai} (Potential periculos)")
print(f"Cazuri unde algoritmul custom a subestimat riscul: {subestimari_safe} (Siguranță maximă)")

if subestimari_ai > 0:
    reducere = 100 * (subestimari_ai - subestimari_safe) / subestimari_ai
    print(f"Eficiența algoritmului de siguranță: Riscul a fost redus cu {reducere:.1f}%")

# 5. GRAFIC COMPARATIV (zona de siguranță)
plt.figure(figsize=(12, 7))

# Extragerere subset de puncte pentru reprezentare grafică
subset = 60
indices = range(subset)

# Plotare subset
plt.plot(indices, y_test[:subset], 'k-o', label='Valoare Reală', linewidth=2, markersize=5)
plt.plot(indices, y_pred_ai[:subset], 'r--', label='Predicție RF Standard', alpha=0.7)
plt.plot(indices, y_pred_safe[:subset], 'g-^', label='Predicție Safety-First', linewidth=2)


# Colorare zonă de siguranță
plt.fill_between(indices, y_pred_ai[:subset], y_pred_safe[:subset], color='green', alpha=0.1,
                 label='Marjă de Siguranță')

plt.title('Algoritm Hibrid "Safety-First" vs RF Standard\n')
plt.xlabel('Eșantioane (exemple din setul de testare)')
plt.ylabel('Inducție Magnetică (µT)')
plt.legend()
plt.grid(True)
plt.tight_layout()

# Salvare și afișare
grafic_safe_path = os.path.join(DIR_GRAFICE, 'Grafic_Safety_First_Testare.png')
plt.savefig(grafic_safe_path)

plt.show(block=True)

# 6. SALVAREA MODELULUI
cale_model_hibrid = os.path.join(DIR_MODELE, 'model_algoritm_custom_Safety_First.joblib')
joblib.dump(custom_algo, cale_model_hibrid)
