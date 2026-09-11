import os
import joblib
import matplotlib.pyplot as plt
import pandas as pd
import shap
import warnings
import tensorflow as tf
from tensorflow.keras.models import load_model

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#  CONFIGURARE DIRECTOARE
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DIR_PROCESATE = os.path.join(ROOT_DIR, '02_Date_Procesate')
DIR_MODELE = os.path.join(ROOT_DIR, '03_Modele_Salvate')
DIR_GRAFICE = os.path.join(ROOT_DIR, '04_Grafice_Rezultate')

print("1. Se încarcă datele")
x_train = joblib.load(os.path.join(DIR_PROCESATE, 'x_train.data'))
x_test = joblib.load(os.path.join(DIR_PROCESATE, 'x_test.data'))

print("2. Se pregătesc eșantioanele")
x_test_esantion = x_test.sample(n=100, random_state=42)
x_train_background = shap.sample(x_train, 25)

# 3. Încărcare scaler și model ANN
scaler = joblib.load(os.path.join(DIR_MODELE, 'scaler_cnn_keras.joblib'))
model = load_model(os.path.join(DIR_MODELE, 'model_cnn_keras.h5'), compile=False)

print("\nÎncepe analiza SHAP pentru 1D-CNN")

# 4. Wrapper pentru scalare și formatare 3D
def predict_cnn_wrapper(x):
    x_scaled = scaler.transform(x)
    x_3d = x_scaled.reshape(x_scaled.shape[0], x_scaled.shape[1], 1)
    return model.predict(x_3d, verbose=0).flatten()

# 5. Configurare SHAP (KernelExplainer)
explainer = shap.KernelExplainer(predict_cnn_wrapper, x_train_background)
shap_values = explainer.shap_values(x_test_esantion)

# 6. Generare și salvare grafic
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, x_test_esantion, show=False)
plt.title('Impactul Caracteristicilor - Model 1D-CNN', fontsize=14, pad=20)
plt.tight_layout()

nume_fisier_grafic = 'Grafic_SHAP_CNN-1D.png'
plt.savefig(os.path.join(DIR_GRAFICE, nume_fisier_grafic), dpi=300, bbox_inches='tight')
plt.close()