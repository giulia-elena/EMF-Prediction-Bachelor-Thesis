import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, mean_absolute_error

# CONFIGURARE DIRECTOARE
DIR_PROCESATE = '02_Date_Procesate'
DIR_MODELE = '03_Modele_Salvate'
DIR_GRAFICE = '04_Grafice_Rezultate'

print("1. Se încarcă modelul optimizat (fără re-antrenare)")
# Încărcăm "creierul" deja antrenat de Grid Search
model_optimizat = joblib.load(os.path.join(DIR_MODELE, 'model_GB_optimizat.joblib'))

print("2. Se încarcă datele de test")
x_test = joblib.load(os.path.join(DIR_PROCESATE, 'x_test.data'))
y_test = joblib.load(os.path.join(DIR_PROCESATE, 'y_test.data'))

print("3. Se generează predicțiile")
predictii = model_optimizat.predict(x_test)

# Recalculăm scorurile doar ca să mă asigur că pe grafic scrie exact ce trebuie
r2 = r2_score(y_test, predictii)
mae = mean_absolute_error(y_test, predictii)
print(f"   Confirmare scoruri: R2={r2:.4f}, MAE={mae:.5f}")

print("4. Se generează și se salvează graficul")
plt.figure(figsize=(10, 6))

sns.scatterplot(x=y_test, y=predictii, alpha=0.5, color='darkgreen', label='Predicții GB Optimizat')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Predicție Ideală')

plt.xlabel('Valori Reale (B [µT])', fontsize=12)
plt.ylabel('Valori Prezise (B [µT])', fontsize=12)
plt.title(f' Performanță Predictivă Gradient Boosting Optimizat\nR² = {r2:.4f} | MAE = {mae:.5f} µT', fontsize=14)
plt.legend()
plt.grid(True, which="both", ls="--", alpha=0.5)

# SALVARE
grafic_path = os.path.join(DIR_GRAFICE, 'Grafic_GB_Optimizat_Performanță.png')
plt.savefig(grafic_path, bbox_inches='tight')

plt.show()