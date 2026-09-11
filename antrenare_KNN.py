import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error

# CONFIGURARE DIRECTOARE
DIR_PROCESATE = '02_Date_Procesate'
DIR_MODELE = '03_Modele_Salvate'
DIR_GRAFICE = '04_Grafice_Rezultate'

# SE CAUTĂ FOLDERUL PENTRU GRAFICE
if not os.path.exists(DIR_GRAFICE):
    os.makedirs(DIR_GRAFICE)

# 1. ÎNCĂRCARE DATE PREPROCESATE DIN FOLDERUL 02_Date_Procesate
print("--- Încărcare date pentru KNN ---")
try:
    X_train = joblib.load(os.path.join(DIR_PROCESATE, 'x_train.data'))
    X_test = joblib.load(os.path.join(DIR_PROCESATE, 'x_test.data'))
    y_train = joblib.load(os.path.join(DIR_PROCESATE, 'y_train.data'))
    y_test = joblib.load(os.path.join(DIR_PROCESATE, 'y_test.data'))
    print(f"Datele au fost încărcate din {DIR_PROCESATE}.")
except FileNotFoundError:
    print(f"EROARE: Fișierele .data nu au fost găsite în {DIR_PROCESATE}.")
    exit()

# 2. SCALARE (obligatorie pentru KNN)
# Scaler-ul transformă datele pentru ca distanțele să fie calculate corect
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 3. ANTRENARE KNN
print("Antrenare K-NN...")
model = KNeighborsRegressor(n_neighbors=10, n_jobs=-1)
model.fit(x_train_scaled, y_train)

# 4. EVALUARE
predictii = model.predict(X_test_scaled)
r2 = r2_score(y_test, predictii)
mae = mean_absolute_error(y_test, predictii)

print(f"\n--- REZULTATE KNN ---")
print(f"Scor de performanță (R2): {r2:.4f}")
print(f"Eroare medie (MAE):    {mae:.5f} µT")

# 5. SALVARE MODEL ȘI SCALER ÎN FOLDERUL 03_Modele_Salvate
# Scaler-ul este necesar pentru interfața grafică pentru că noile date trebuie scalate la fel
joblib.dump(model, os.path.join(DIR_MODELE, 'model_KNN.joblib'))
joblib.dump(scaler, os.path.join(DIR_MODELE, 'scaler_KNN.joblib'))

# 6. VIZUALIZARE GRAFICĂ (ACTUAL VS PREDICTED)
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test, y=predictii, alpha=0.3, color='purple', label='Predicții KNN')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Ideal')

plt.xlabel('Valori Reale (B [µT])')
plt.ylabel('Valori Prezise (B [µT])')
plt.title(f'Performanță Predictivă KNN\n(R2 = {r2:.4f}, MAE = {mae:.5f} µT)')
plt.legend()
plt.grid(True)

# SALVARE GRAFIC ÎN FOLDERUL 04_Grafice_Rezultate
grafic_path = os.path.join(DIR_GRAFICE, 'Grafic_KNN_Performanță.png')
plt.savefig(grafic_path)
plt.show()