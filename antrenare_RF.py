
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error

# CONFIGURARE DIRECTOARE
DIR_PROCESATE = '02_Date_Procesate'
DIR_MODELE = '03_Modele_Salvate'
DIR_GRAFICE = '04_Grafice_Rezultate'

# SE CAUTĂ FOLDERUL PENTRU GRAFICE
if not os.path.exists(DIR_GRAFICE):
    os.makedirs(DIR_GRAFICE)

# 1. ÎNCĂRCARE DATE PREPROCESATE DIN FOLDERUL 02_Date_Procesate
print("--- Încărcare date pentru Random Forest ---")
try:
    x_train = joblib.load(os.path.join(DIR_PROCESATE, 'x_train.data'))
    x_test = joblib.load(os.path.join(DIR_PROCESATE, 'x_test.data'))
    y_train = joblib.load(os.path.join(DIR_PROCESATE, 'y_train.data'))
    y_test = joblib.load(os.path.join(DIR_PROCESATE, 'y_test.data'))
    print(f"Datele au fost încărcate din {DIR_PROCESATE}.")
except FileNotFoundError:
    print(f"EROARE: Fișierele .data nu au fost găsite în {DIR_PROCESATE}.")
    exit()

# 2. ANTRENARRE MODEL
print("Antrenare Random Forest...")
model = RandomForestRegressor(n_estimators=300, min_samples_leaf=5, random_state=42, n_jobs=-1)
model.fit(x_train, y_train)

# 3. EVALUARE PERFORMANȚĂ
predictii = model.predict(x_test)
r2 = r2_score(y_test, predictii)
mae = mean_absolute_error(y_test, predictii)

print(f"\n--- REZULTATE RANDOM FOREST ---")
print(f"Scor de PERFORMANȚĂ (R2): {r2:.4f}")
print(f"Eroare medie (MAE):    {mae:.5f} µT")

# 4. IMPORTANȚA CARACTERISTICILOR (PROCENTE)
print(f"\n--- IMPORTANȚA CARACTERISTICILOR ---")
importances = pd.DataFrame({
    'Caracteristica': x_train.columns,
    'Importanta (%)': model.feature_importances_ * 100
}).sort_values(by='Importanta (%)', ascending=False)

print(importances.to_string(index=False))

# 5. SALVARE MODEL FINAL ÎN FOLDERUL 03_Modele_Salvate
model_path = os.path.join(DIR_MODELE, 'model_RF.joblib')
joblib.dump(model, model_path)

# 6. VIZUALIZARE GRAFICĂ (ACTUAL VS PREDICTED)
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test, y=predictii, alpha=0.3, color='blue', label='Valori înregistrate')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Predicție Ideală')

plt.xlabel('Valori Reale (B [µT])')
plt.ylabel('Valori Prezise (B [µT])')
plt.title(f'Performanță Predictivă Random Forest\n(R2 = {r2:.4f}, MAE = {mae:.5f} µT)')
plt.legend()
plt.grid(True)

# SALVARE GRAFIC ÎN FOLDERUL 04_Grafice_Rezultate
grafic_path = os.path.join(DIR_GRAFICE, 'Grafic_RF_Performanță.png')
plt.savefig(grafic_path)
plt.show()



# # Utilizare Grid Search, dar fără rezultate deosebite
# import joblib
# import os
# import pandas as pd
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.model_selection import GridSearchCV
# from sklearn.metrics import r2_score, mean_absolute_error
#
# #  CONFIGURARE DIRECTOARE
# DIR_PROCESATE = '02_Date_Procesate'
# DIR_MODELE = '03_Modele_Salvate'
#
# x_train = joblib.load(os.path.join(DIR_PROCESATE, 'x_train.data'))
# y_train = joblib.load(os.path.join(DIR_PROCESATE, 'y_train.data'))
# x_test = joblib.load(os.path.join(DIR_PROCESATE, 'x_test.data'))
# y_test = joblib.load(os.path.join(DIR_PROCESATE, 'y_test.data'))
#
# # 1. DEFINIREA GRILEI DE PARAMETRI PENTRU RF
# # RF preferă arbori mai adânci decât GB, așa că pun și opțiunea "None" (cresc cât vor)
# param_grid = {
#     'n_estimators': [100, 200, 300],          # câți arbori votanți sunt
#     'max_depth': [None, 10, 20],              # Cât de adânc merg ramificațiile
#     'min_samples_split': [2, 5, 10],          # Puncte minime pentru a face o tăietură
#     'min_samples_leaf': [2, 4, 6]             # Știu că 5 a fost bun, așa că testez zona asta
# }
#
# # 2. CONFIGURAREA GRID SEARCH
# # n_jobs=-1 ca să folosească tot procesorul și să se miște rapid
# rf_base = RandomForestRegressor(random_state=42, n_jobs=-1)
#
# grid_search = GridSearchCV(
#     estimator=rf_base,
#     param_grid=param_grid,
#     cv=3,                 # Testează pe 3 bucăți de date diferite (Cross-Validation)
#     scoring='r2',
#     verbose=1
# )
#
# # 3. EXECUTARE CĂUTARE
# grid_search.fit(x_train, y_train)
#
# # 4. EXTRAGEREA CELUI MAI BUN MODEL
# best_rf_model = grid_search.best_estimator_
# print("\n--- REZULTATE GRID SEARCH RANDOM FOREST ---")
# print(f"Cei mai buni parametri găsiți:\n{grid_search.best_params_}")
#
# # 5. EVALUARE FINALĂ
# predictii = best_rf_model.predict(x_test)
# r2_final = r2_score(y_test, predictii)
# mae_final = mean_absolute_error(y_test, predictii)
#
# print(f"\nPerformanță RF Optimizat:")
# print(f"Scor R2: {r2_final:.4f}")
# print(f"Scor MAE: {mae_final:.5f} µT")
#
# # 6. SALVAREA MODELULUI OPTIMIZAT
# joblib.dump(best_rf_model, os.path.join(DIR_MODELE, 'model_RF_optimizat.joblib'))
#
#
# # Cei mai buni parametri găsiți
# # {'max_depth': None, 'min_samples_leaf': 6, 'min_samples_split': 2, 'n_estimators': 300}
# #
# # Performanță RF optimizat:
# # Scor R2: 0.9456
# # Scor MAE: 0.00172 µT