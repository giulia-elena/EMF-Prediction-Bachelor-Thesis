import joblib
import pandas as pd
import os
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score, mean_absolute_error

# CONFIGURARE DIRECTOARE
DIR_PROCESATE = '02_Date_Procesate'
DIR_MODELE = '03_Modele_Salvate'

# 1. ÎNCĂRCARE DATE
print("--- Încărcare date pentru Grid Search (Gradient Boosting) ---")
x_train = joblib.load(os.path.join(DIR_PROCESATE, 'x_train.data'))
y_train = joblib.load(os.path.join(DIR_PROCESATE, 'y_train.data'))
x_test = joblib.load(os.path.join(DIR_PROCESATE, 'x_test.data'))
y_test = joblib.load(os.path.join(DIR_PROCESATE, 'y_test.data'))

# 2. DEFINIREA GRILEI DE PARAMETRI
# Testează diferite combinații pentru a vedea care e cea mai bună
param_grid = {
    'n_estimators': [100, 200, 300],           # Numărul de arbori
    'learning_rate': [0.01, 0.05, 0.1, 0.2],   # Cât de repede învață din erori
    'max_depth': [3, 4, 5, 6],                 # Adâncimea arborilor
    'min_samples_leaf': [3, 5, 7]              # Numărul minim de puncte într-o frunză
}

# 3. CONFIGURAREA GRID SEARCH
print("Începe căutarea (Grid Search)...")
gb_base = GradientBoostingRegressor(random_state=42)

# cv=3 înseamnă că testează fiecare variantă pe 3 bucăți diferite de date (Cross-Validation)
grid_search = GridSearchCV(
    estimator=gb_base,
    param_grid=param_grid,
    cv=3,
    scoring='r2',
    n_jobs=-1,
    verbose=1
)

# 4. EXECUTARE CĂUTARE
grid_search.fit(x_train, y_train)

# 5. EXTRAGEREA CELUI MAI BUN MODEL
best_model = grid_search.best_estimator_
print("\n--- REZULTATE GRID SEARCH ---")
print(f"Cei mai buni parametri găsiți:\n{grid_search.best_params_}")

# 6. EVALUARE FINALĂ
predictii = best_model.predict(x_test)
r2_final = r2_score(y_test, predictii)
mae_final = mean_absolute_error(y_test, predictii)

print(f"\nPerformanță optimizată:")
print(f"Scor de performanță (R2): {r2_final:.4f}")
print(f"Eroare medie (MAE): {mae_final:.5f} µT")

# 7. SALVAREA MODELULUI OPTIMIZAT
joblib.dump(best_model, os.path.join(DIR_MODELE, 'model_GB_optimizat.joblib'))

