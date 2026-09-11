
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# --- CONFIGURARE DIRECTOARE ---
DIR_PROCESATE = '02_Date_Procesate'
DIR_MODELE = '03_Modele_Salvate'

# Cautare foldere
for folder in [DIR_PROCESATE, DIR_MODELE]:
    if not os.path.exists(folder):
        os.makedirs(folder)

# 1. Încărcare date
print("--- Începerea etapei de prelucrare a datelor ---")
# Citesc Preprocesare.csv din folderul 02_Date_Procesate
df = pd.read_csv(os.path.join(DIR_PROCESATE, 'Preprocesare.csv'))

# 2. Curățare inițială (NaN)
cols_esentiale = ['Frecventa', 'Distanta', 'Sursa_Echipament', 'Zona_Tren', 'Locatie', 'Tip_Masurare', 'Total_B']
df_clean = df.dropna(subset=cols_esentiale).copy()

# 3. Identificare și salvare outliers (metoda IQR)
Q1 = df_clean['Total_B'].quantile(0.25)
Q3 = df_clean['Total_B'].quantile(0.75)
IQR = Q3 - Q1
limita_inf = Q1 - 1.5 * IQR
limita_sup = Q3 + 1.5 * IQR

# Separare date de outliers
df_final = df_clean[(df_clean['Total_B'] >= limita_inf) & (df_clean['Total_B'] <= limita_sup)].copy()
outliers = df_clean[(df_clean['Total_B'] < limita_inf) | (df_clean['Total_B'] > limita_sup)].copy()

# Salvare outliers în folderul 02_Date_Procesate
outliers.to_csv(os.path.join(DIR_PROCESATE, 'OUTLIERS.csv'), index=False)

print(f"\n--- ANALIZĂ OUTLIERS ---")
if not outliers.empty:
    print(f"Număr total outliers detectați: {len(outliers)}")
    print(f"Cel mai MARE outlier înregistrat: {outliers['Total_B'].max():.4f} µT")
    print(f"Cel mai MIC outlier înregistrat:  {outliers['Total_B'].min():.4f} µT")
else:
    print("Nu au fost detectați outliers.")

# 4. Encoding
le_sursa, le_zona, le_locatie, le_tip = LabelEncoder(), LabelEncoder(), LabelEncoder(), LabelEncoder()

df_final.loc[:, 'Sursa_Encoded'] = le_sursa.fit_transform(df_final['Sursa_Echipament'])
df_final.loc[:, 'Zona_Encoded'] = le_zona.fit_transform(df_final['Zona_Tren'])
df_final.loc[:, 'Locatie_Encoded'] = le_locatie.fit_transform(df_final['Locatie'])
df_final.loc[:, 'Tip_Encoded'] = le_tip.fit_transform(df_final['Tip_Masurare'])

# Salvare encodere în folderul 03_Modele_Salvate
joblib.dump({'sursa': le_sursa, 'zona': le_zona, 'locatie': le_locatie, 'tip': le_tip},
            os.path.join(DIR_MODELE, 'encoders.joblib'))




# 5. Definire X și Y
features = ['Frecventa', 'Distanta', 'Sursa_Encoded', 'Zona_Encoded', 'Locatie_Encoded', 'Tip_Encoded']
x = df_final[features]
y = df_final['Total_B']

# 6. Split train/val/test (70% train, 15% validation, 15% test)
# Extrag prima dată 15% pentru test
x_temp, x_test, y_temp, y_test = train_test_split(x, y, test_size=0.15, random_state=42)

# Din cei 85% rămași, extrag ~17.647% pentru a obține 15% din total pentru validare
x_train, x_val, y_train, y_val = train_test_split(x_temp, y_temp, test_size=0.17647, random_state=42)

# 7. Salvare date preprocesate în folderul 02_Date_Procesate
joblib.dump(x_train, os.path.join(DIR_PROCESATE, 'x_train.data'))
joblib.dump(y_train, os.path.join(DIR_PROCESATE, 'y_train.data'))

joblib.dump(x_val, os.path.join(DIR_PROCESATE, 'x_val.data'))
joblib.dump(y_val, os.path.join(DIR_PROCESATE, 'y_val.data'))

joblib.dump(x_test, os.path.join(DIR_PROCESATE, 'x_test.data'))
joblib.dump(y_test, os.path.join(DIR_PROCESATE, 'y_test.data'))

print(f"\n--- Rezultat final ---")
print(f"Total rânduri: {len(df_final)}")
print(f"Set antrenare  : {len(x_train)} rânduri (~70%)")
print(f"Set validare     : {len(x_val)} rânduri (~15%)")
print(f"Set testare     : {len(x_test)} rânduri (~15%)")


