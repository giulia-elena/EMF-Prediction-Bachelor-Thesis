import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, mean_absolute_error

# CONFIGURARE DIRECTOARE
DIR_PROCESATE = '02_Date_Procesate'
DIR_MODELE = '03_Modele_Salvate'
DIR_GRAFICE = '04_Grafice_Rezultate'

# SE CAUTĂ FOLDERUL PENTRU GRAFICE
if not os.path.exists(DIR_GRAFICE):
    os.makedirs(DIR_GRAFICE)

# 1. ÎNCĂRCARE DATE PREPROCESATE DIN FOLDERUL 02_Date_Procesate
print("--- Încărcare date pentru Decision Tree ---")
try:
    x_train = joblib.load(os.path.join(DIR_PROCESATE, 'x_train.data'))
    x_test = joblib.load(os.path.join(DIR_PROCESATE, 'x_test.data'))
    y_train = joblib.load(os.path.join(DIR_PROCESATE, 'y_train.data'))
    y_test = joblib.load(os.path.join(DIR_PROCESATE, 'y_test.data'))
    print(f"Datele au fost încărcate din {DIR_PROCESATE}.")
except FileNotFoundError:
    print(f"EROARE: Fișierele .data nu au fost găsite în {DIR_PROCESATE}.")
    exit()

# 2. ANTRENARE MODEL
# Se consideră două cazuri: max_depth=3 pentru a obține scorul de ~83% sau max_depth=12 pentru a obține 93%
print("Antrenare DT (max_depth=12)...")
model = DecisionTreeRegressor(max_depth=12, random_state=42)
model.fit(x_train, y_train)

# 3. EVALUARE PERFORMANȚĂ
predictii = model.predict(x_test)
r2 = r2_score(y_test, predictii)
mae = mean_absolute_error(y_test, predictii)

print(f"\n--- REZULTATE DT ---")
print(f"Scor de performanță (R2): {r2:.4f}")
print(f"Eroare medie (MAE):    {mae:.5f} µT")

# 4. IMPORTANȚA CARACTERISTICILOR
print(f"\n--- IMPORTANȚA CARACTERISTICILOR ---")
importances = pd.DataFrame({
    'Caracteristica': x_train.columns,
    'Importanta (%)': model.feature_importances_ * 100
}).sort_values(by='Importanta (%)', ascending=False)
print(importances.to_string(index=False))

# 5. SALVARE MODEL FINAL ÎN FOLDERUL 03_Modele_Salvate
model_path = os.path.join(DIR_MODELE, 'model_DT.joblib')
joblib.dump(model, model_path)
print(f"\n[INFO] Modelul a fost salvat în: {model_path}")

# 6. VIZUALIZARE GRAFICĂ (ACTUAL VS PREDICTED)
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test, y=predictii, alpha=0.3, color='orange', label='Predicții DT')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Ideal')

plt.xlabel('Valori Reale (B [µT])')
plt.ylabel('Valori Prezise (B [µT])')
plt.title(f'Performanță Predictivă Decision Tree\n(R2 = {r2:.4f}, MAE = {mae:.5f} µT)')
plt.legend()
plt.grid(True)

# SALVARE GRAFIC ÎN FOLDERUL 04_Grafice_Rezultate
grafic_path = os.path.join(DIR_GRAFICE, 'Grafic_DT_Performanță_3.png')
plt.savefig(grafic_path)
plt.show()