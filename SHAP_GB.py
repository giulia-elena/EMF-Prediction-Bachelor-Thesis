import os
import joblib
import matplotlib.pyplot as plt
import pandas as pd
import shap
import warnings

warnings.filterwarnings('ignore')

#  CONFIGURARE DIRECTOARE
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DIR_PROCESATE = os.path.join(ROOT_DIR, '02_Date_Procesate')
DIR_MODELE = os.path.join(ROOT_DIR, '03_Modele_Salvate')
DIR_GRAFICE = os.path.join(ROOT_DIR, '04_Grafice_Rezultate')

# 1. Încărcare Date
print("1. Se încarcă datele")
x_train = joblib.load(os.path.join(DIR_PROCESATE, 'x_train.data'))
x_test = joblib.load(os.path.join(DIR_PROCESATE, 'x_test.data'))

print("2. Se pregătește eșantionul (100 inregistrări)")
x_test_esantion = x_test.sample(n=100, random_state=42)

# 3. Încărcare model
cale_model = os.path.join(DIR_MODELE, 'model_GB_Optimizat.joblib')
model = joblib.load(cale_model)

print("\nÎncepe analiza SHAP pentru GB")
# 4. Configurare SHAP (TreeExplainer)
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(x_test_esantion, check_additivity=False)

# 5. Generare și salvare grafic
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, x_test_esantion, show=False)
plt.title('Impactul Caracteristicilor - Model Gradient Boosting Optimizat', fontsize=14, pad=20)
plt.tight_layout()

nume_fisier_grafic = 'Grafic_SHAP_GB_Optimizat.png'
plt.savefig(os.path.join(DIR_GRAFICE, nume_fisier_grafic), dpi=300, bbox_inches='tight')
plt.close()