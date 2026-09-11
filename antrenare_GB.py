
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error

# CONFIGURARE DIRECTOARE
DIR_PROCESATE = '02_Date_Procesate'
DIR_GRAFICE = '04_Grafice_Rezultate'
DIR_MODELE = '03_Modele_Salvate'

# 1. ÎNCĂRCARE DATE PREPROCESATE DIN FOLDERUL 02_Date_Procesate
print("--- Încărcare date pentru Gradient Boosting ---")
x_train = joblib.load(os.path.join(DIR_PROCESATE, 'x_train.data'))
y_train = joblib.load(os.path.join(DIR_PROCESATE, 'y_train.data'))
x_test = joblib.load(os.path.join(DIR_PROCESATE, 'x_test.data'))
y_test = joblib.load(os.path.join(DIR_PROCESATE, 'y_test.data'))

# 2. DEFINIRE ȘI ANTRENARE MODEL CU PARAMETRII DE LA RANDOM FOREST
print("Antrenare model GB cu parametrii: n_estimators=300, min_samples_leaf=5...")
# learning_rate = 0.1 automat din biblioteca
gb_test = GradientBoostingRegressor(
    n_estimators=300,
    min_samples_leaf=5,
    random_state=42
)
gb_test.fit(x_train, y_train)

# 3. EVALUARE PERFORMANȚĂ
predictii_test = gb_test.predict(x_test)
r2_test = r2_score(y_test, predictii_test)
mae_test = mean_absolute_error(y_test, predictii_test)

print(f"\n--- REZULTATE GRADIENT BOOSTING ---")
print(f"Scor de performanță (R2): {r2_test:.4f}")
print(f"Eroare medie (MAE): {mae_test:.5f} µT")

# 4. VIZUALIZARE GRAFICĂ (ACTUAL VS PREDICTED)
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test, y=predictii_test, alpha=0.3, color='blue', label='Valori înregistrate')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Predicție Ideală')

plt.xlabel('Valori Reale (B [µT])')
plt.ylabel('Valori Prezise (B [µT])')
plt.title(f'Performanță Predictivă Gradient Boosting\n(R2 = {r2_test:.4f}, MAE = {mae_test:.5f} µT)')
plt.legend()
plt.grid(True)

# SALVARE GRAFIC ÎN FOLDERUL 04_Grafice_Rezultate
grafic_path = os.path.join(DIR_GRAFICE, 'Grafic_GB_Performanță.png')
plt.savefig(grafic_path)
plt.show()

joblib.dump(gb_test, os.path.join(DIR_MODELE, 'model_GB_referinta.joblib'))