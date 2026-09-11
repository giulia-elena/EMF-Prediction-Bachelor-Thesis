import os
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tensorflow.keras import models, layers
from tensorflow.keras.callbacks import EarlyStopping

# CONFIGURARE DIRECTOARE
DIR_PROCESATE = '02_Date_Procesate'
DIR_MODELE = '03_Modele_Salvate'
DIR_GRAFICE = '04_Grafice_Rezultate'
os.makedirs(DIR_GRAFICE, exist_ok=True)

# 1. ÎNCĂRCARE DATE
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

# Salvare scaler
joblib.dump(scaler, os.path.join(DIR_MODELE, 'scaler_cnn_keras.joblib'))

# 3. RESHAPE PENTRU 1D CNN
# CNN are nevoie de 3 dimensiuni: (rânduri, caracteristici, canale)
n_features = x_train_scaled.shape[1]
x_train_cnn = x_train_scaled.reshape(x_train_scaled.shape[0], n_features, 1)
x_val_cnn = x_val_scaled.reshape(x_val_scaled.shape[0], n_features, 1)
x_test_cnn = x_test_scaled.reshape(x_test_scaled.shape[0], n_features, 1)

# 4. DEFINIREA MODELULUI (CNN + ANN)
model = models.Sequential()

# Partea de extracție a trăsăturilor (CNN)
# Se folosește kernel_size=1 sau 2 pentru date tabelare
model.add(layers.Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(n_features, 1)))
# MaxPooling tăia din date, îl scot pentru a nu pierde informație utilă din tabel
# model.add(layers.MaxPooling1D(pool_size=2))

# Trecerea la 1D
model.add(layers.Flatten())

# Partea de decizie (aliniată cu arhitectura optimă de la ANN: 64-32)
model.add(layers.Dense(64, kernel_initializer='normal', activation='relu'))
model.add(layers.Dense(32, kernel_initializer='normal', activation='relu'))

# Ieșirea
model.add(layers.Dense(1, kernel_initializer='normal', activation='linear'))

model.compile(loss='mse', optimizer='adam', metrics=['mae'])

# 5. ANTRENAREA MODELULUI
early_stop = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True, verbose=1)

print("\nÎncepe antrenarea rețelei convoluționale (CNN 1D)...")
istoric = model.fit(
    x_train_cnn, y_train,
    validation_data=(x_val_cnn, y_val),
    epochs=300,
    batch_size=200, # Am luat batch_size-ul bun de la ANN
    callbacks=[early_stop],
    verbose=0
)

# Salvăm modelul
model.save(os.path.join(DIR_MODELE, 'model_cnn_keras.h5'))

# 6. EVALUARE ȘI GRAFIC PE SETUL DE VALIDARE
predictii_val_cnn = model.predict(x_val_cnn).flatten()

r2_val = r2_score(y_val, predictii_val_cnn)
mae_val = mean_absolute_error(y_val, predictii_val_cnn)

print(f"\n--- REZULTATE VALIDARE 1D CNN ---")
print(f"Scor de performanță (R2): {r2_val:.4f}")
print(f"Eroare medie (MAE):     {mae_val:.5f} µT")


# --- Creare Figură cu 2 subgrafice (1 rând, 2 coloane) ---
plt.figure(figsize=(14, 6))

# Graficul 1: Curba de învățare (Loss)
plt.subplot(1, 2, 1)
plt.plot(istoric.history['loss'], label='Train Loss', color='blue')
plt.plot(istoric.history['val_loss'], label='Val Loss', color='red')
plt.title('Evoluția Învățării 1D CNN (Loss / MSE)')
plt.xlabel('Epoca')
plt.ylabel('Eroare (MSE)')
plt.legend()
plt.grid(True)

# Graficul 2: Scatter Plot cu predicțiile pe Validare
plt.subplot(1, 2, 2)
plt.scatter(y_val, predictii_val_cnn, alpha=0.5, color='orange', label='Predicții 1D CNN (Validare)')
min_val_v = min(y_val.min(), predictii_val_cnn.min())
max_val_v = max(y_val.max(), predictii_val_cnn.max())
plt.plot([min_val_v, max_val_v], [min_val_v, max_val_v], 'r--', lw=2, label='Linia Ideală')
plt.title(f'Performanță Validare 1D CNN\nR² = {r2_val:.4f} | MAE = {mae_val:.5f} µT')
plt.xlabel('Valori Reale (B [µT])')
plt.ylabel('Valori Prezise 1D CNN (B [µT])')
plt.legend()
plt.grid(True)

plt.tight_layout() # Previne suprapunerea textelor de pe axe

# SALVARE GRAFIC (curbă + scatter)
plt.savefig(os.path.join(DIR_GRAFICE, 'Grafic_CNN_Validare.png'), bbox_inches='tight')
plt.show()

# # 7. EVALUARE ȘI GRAFIC PE SETUL DE TEST  (comentat pentru sigurață)
# predictii_cnn = model.predict(x_test_cnn).flatten()
#
# r2_test = r2_score(y_test, predictii_cnn)
# mae_test = mean_absolute_error(y_test, predictii_cnn)
#
# print(f"\n--- REZULTATE FINALE 1D CNN ---")
# print(f"Scor de performanță (R2): {r2_test:.4f}")
# print(f"Eroare medie (MAE):     {mae_test:.5f} µT")
#
# plt.figure(figsize=(10, 6))
# sns.scatterplot(x=y_test, y=predictii_cnn, alpha=0.5, color='purple', label='Predicții CNN (Testare)')
# min_val_t = min(y_test.min(), predictii_cnn.min())
# max_val_t = max(y_test.max(), predictii_cnn.max())
# plt.plot([min_val_t, max_val_t], [min_val_t, max_val_t], 'r--', lw=2, label='Predicție Ideală')
#
# plt.xlabel('Inducție Măsurată Real (µT)', fontsize=12)
# plt.ylabel('Inducție Prezisă de Rețea (µT)', fontsize=12)
# plt.title(f'Performanță Finală 1D CNN pe Setul de Test\nR² = {r2_test:.4f} | MAE = {mae_test:.5f} µT', fontsize=14)
# plt.legend()
# plt.grid(True, which="both", ls="--", alpha=0.5)
#
# # SALVARE GRAFIC
# plt.savefig(os.path.join(DIR_GRAFICE, 'Grafic_CNN_Testare.png'), bbox_inches='tight')
# plt.show()
#
