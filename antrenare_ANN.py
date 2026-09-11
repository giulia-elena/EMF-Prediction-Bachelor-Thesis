import os
import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score

# Importare TensorFlow / Keras
from tensorflow.keras import models, layers
from tensorflow.keras.callbacks import EarlyStopping

# CONFIGURARE DIRECTOARE
DIR_PROCESATE = '02_Date_Procesate'
DIR_MODELE = '03_Modele_Salvate'
DIR_GRAFICE = '04_Grafice_Rezultate'
os.makedirs(DIR_GRAFICE, exist_ok=True)

# 1. ÎNCĂRCARE DATE
print("Se încarcă datele")
x_train = joblib.load(os.path.join(DIR_PROCESATE, 'x_train.data'))
y_train = joblib.load(os.path.join(DIR_PROCESATE, 'y_train.data'))

x_val = joblib.load(os.path.join(DIR_PROCESATE, 'x_val.data'))
y_val = joblib.load(os.path.join(DIR_PROCESATE, 'y_val.data'))

x_test = joblib.load(os.path.join(DIR_PROCESATE, 'x_test.data'))
y_test = joblib.load(os.path.join(DIR_PROCESATE, 'y_test.data'))

# 2. SCALAREA DATELOR
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_val_scaled = scaler.transform(x_val)
x_test_scaled = scaler.transform(x_test)

# Salvare scaler pentru interfața grafică
joblib.dump(scaler, os.path.join(DIR_MODELE, 'scaler_ann_keras.joblib'))


# 3. DEFINIREA MODELULUI
def construieste_model_ann(num_features):
    model = models.Sequential()
    # Încercarea 1, model simplu
    # Stratul ascuns 1 (64 neuroni)
    model.add(layers.Dense(64, input_dim=num_features, kernel_initializer='normal', activation='relu'))
    # Stratul ascuns 2 (32 neuroni)
    model.add(layers.Dense(32, kernel_initializer='normal', activation='relu'))
    # Stratul de ieșire (1 singur neuron pentru predicția valorii B, activare liniară)
    model.add(layers.Dense(1, kernel_initializer='normal', activation='linear'))

    # Încercarea 2, model mai complex
    # # Stratul 1 (mai lat)
    # model.add(layers.Dense(128, input_dim=num_features, kernel_initializer='normal', activation='relu'))
    # # Stratul 2 (mai lat)
    # model.add(layers.Dense(64, kernel_initializer='normal', activation='relu'))
    # # Stratul 3 (nou - ajută la rafinare)
    # model.add(layers.Dense(32, kernel_initializer='normal', activation='relu'))
    # # Ieșirea
    # model.add(layers.Dense(1, kernel_initializer='normal', activation='linear'))

    # Compilare
    model.compile(loss='mse', optimizer='adam', metrics=['mae'])
    return model


# 4. ANTRENAREA MODELULUI
numar_caracteristici = x_train_scaled.shape[1]
model_keras = construieste_model_ann(numar_caracteristici)

# Early Stopping (se oprește dacă eroarea pe validare nu mai scade 20 de epoci la rând)
early_stop = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True, verbose=1)

print("\nÎncepe antrenarea rețelei neuronale")
istoric = model_keras.fit(
    x_train_scaled, y_train,
    validation_data=(x_val_scaled, y_val),  # Aici folosește setul de validare
    epochs=300,  # Numărul maxim de treceri prin date
    batch_size=200,  # Ia câte 200 de rânduri odată
    callbacks=[early_stop],
    verbose=1  # Afișează progresul
)

# Se salvează modelul antrenat (format Keras)
model_keras.save(os.path.join(DIR_MODELE, 'model_ann_keras_referinta.h5'))

# ==========================================================
# 5. EVALUARE PE SETUL DE VALIDARE
# ==========================================================

print("\n--- EVALUARE PE SETUL DE VALIDARE ---")
y_pred_val = model_keras.predict(x_val_scaled).flatten()

r2_val = r2_score(y_val, y_pred_val)
mae_val = mean_absolute_error(y_val, y_pred_val)

print(f"Scor de performanță (R2) (validare): {r2_val:.4f}")
print(f"Eroare medie (MAE) (validare): {mae_val:.5f} µT")

# 6. GENERARE GRAFICE (validare)
plt.figure(figsize=(12, 5))

# Graficul 1: Curba de învățare (Loss)
plt.subplot(1, 2, 1)
plt.plot(istoric.history['loss'], label='Train Loss')
plt.plot(istoric.history['val_loss'], label='Val Loss')
plt.title('Evoluția Învățării (Loss / MSE)')
plt.xlabel('Epoca')
plt.ylabel('Eroare (MSE)')
plt.legend()
plt.grid(True)

# Graficul 2: Scatter Plot cu predicțiile pe validare
plt.subplot(1, 2, 2)
plt.scatter(y_val, y_pred_val, alpha=0.5, color='green', label='Predicții (Validare)')
plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'r--', lw=2, label='Linia Ideală')
plt.title(f'Performanță (Validare)\nR² = {r2_val:.4f} | MAE = {mae_val:.5f} µT')
plt.xlabel('Inducție Măsurată (µT)')
plt.ylabel('Inducție Prezisă (µT)')
plt.legend()
plt.grid(True)

plt.tight_layout()

# SALVARE GRAFIC
grafic_path = os.path.join(DIR_GRAFICE, 'Grafic_ANN_Validare.png')
plt.savefig(grafic_path)
plt.show()

# # ==========================================================
# # 7. TESTUL FINAL ȘI GRAFICUL DE TESTARE (comentat pentru siguranță)
# # ==========================================================
# print("\n--- EVALUARE FINALĂ PE SETUL DE TEST (15% DATE) ---")
#
# # Predicțiile pe datele de test
# y_pred_test = model_keras.predict(x_test_scaled).flatten()
#
# # Scorurile finale
# r2_test = r2_score(y_test, y_pred_test)
# mae_test = mean_absolute_error(y_test, y_pred_test)
#
# print(f"Scor de performanță (R2) (test): {r2_test:.4f}")
# print(f"Eroare medie (MAE) (test): {mae_test:.5f} µT")
#
# # Generare grafic pentru setul de test
# plt.figure(figsize=(8, 6))
# plt.scatter(y_test, y_pred_test, alpha=0.6, color='purple', label='Predicții (Test Set)')
# plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Linia Ideală')
#
# plt.title(f'Performanță ANN pe Setul de Test\nR² = {r2_test:.4f} | MAE = {mae_test:.5f} µT')
# plt.xlabel('Inducție Măsurată (µT)')
# plt.ylabel('Inducție Prezisă (µT)')
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
#
# # Salvare grafic
# grafic_test_path = os.path.join(DIR_GRAFICE, 'Grafic_ANN_Testare.png')
# plt.savefig(grafic_test_path)
# print(f"[INFO] Graficul final de test a fost salvat în: {grafic_test_path}")
#
# plt.show(block=True)