import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import os

# CONFIGURARE DIRECTOARE
DIR_PROCESATE = '02_Date_Procesate'
DIR_GRAFICE = '04_Grafice_Rezultate'

# 1. Încărcare set de date (folosind x_train, deoarece pe el a învățat modelul)
print("Se încarcă datele pentru Heatmap")
x_train = joblib.load(os.path.join(DIR_PROCESATE, 'x_train.data'))

# 2. Calculare matrice de corelație
# Măsoară relația matematică între coloane (de la -1 la 1)
matrice_corelatie = x_train.corr()

# 3. Desenare Heatmap
plt.figure(figsize=(12, 8))

grafic = sns.heatmap(
    matrice_corelatie,
    annot=True,    # pune și numerele în căsuțe
    cmap='RdPu',   # Red-Purple - generează un gradient roz
    fmt=".2f",     # lasă doar două zecimale
    linewidths=1,
    linecolor='white',
    cbar_kws={'label': 'Nivel de corelație'}
)

plt.title('Matricea de Corelație a Caracteristicilor', fontsize=16, pad=20)

# Rotesc un pic etichetele ca să se citească mai ușor
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.yticks(rotation=0, fontsize=10)
plt.tight_layout()

# 4. Salvare imagine în folderul de grafice
grafic_path = os.path.join(DIR_GRAFICE, 'Grafic_Heatmap_Roz.png')
plt.savefig(grafic_path, dpi=300) # dpi=300 o face HD
plt.show()