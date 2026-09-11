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

print("2. Se pregătesc eșantioanele")
# Se limitează datele pentru că algoritmul KernelExplainer are un cost computațional mare
x_test_esantion = x_test.sample(n=100, random_state=42)
x_train_background = shap.sample(x_train, 25)

# 3. Încărcare model KNN și scaler
cale_model = os.path.join(DIR_MODELE, 'model_KNN.joblib')
cale_scaler = os.path.join(DIR_MODELE, 'scaler_cnn_keras.joblib')

print("\nÎncepe analiza SHAP pentru KNN")

if not os.path.exists(cale_model):
    print("EROARE: Nu s-a găsit modelul KNN")
else:
    model = joblib.load(cale_model)
    scaler = joblib.load(cale_scaler)


    # 4. Wrapper pentru scalare
    # SHAP trimite date nescalate, le scalez doar pentru predicție
    def predict_knn_wrapper(x):
        x_scaled = scaler.transform(x)
        return model.predict(x_scaled)


    # 5. Configurare SHAP (KernelExplainer pentru modele bazate pe distanțe)
    explainer = shap.KernelExplainer(predict_knn_wrapper, x_train_background)
    shap_values = explainer.shap_values(x_test_esantion)

    # 6. Generare și salvare grafic
    plt.figure(figsize=(10, 6))

    shap.summary_plot(shap_values, x_test_esantion, show=False)
    plt.title('Impactul Caracteristicilor - Model k-NN', fontsize=14, pad=20)
    plt.tight_layout()

    nume_fisier_grafic = 'Grafic_SHAP_KNN.png'
    plt.savefig(os.path.join(DIR_GRAFICE, nume_fisier_grafic), dpi=300, bbox_inches='tight')
    plt.close()
