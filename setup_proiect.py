import os
import shutil

def organizeaza_proiectul():
    # 1. Se definește structura de foldere
    foldere = {
        '01_Date_Brute': ['.xlsx', '.xls', 'dictionar.csv'],
        '02_Date_Procesate': ['.csv', '.data'],
        '03_Modele_Salvate': ['.joblib'],
        '04_Grafice_Rezultate': ['.png', '.jpg']
    }

    # 2. Se crează folderele
    for folder in foldere:
        if not os.path.exists(folder):
            os.makedirs(folder)
            print(f"Creat folder: {folder}")

    # 3. Se mutăm fișierele existente (în caz că nu se află în folderele corecte)
    files = [f for f in os.listdir('.') if os.path.isfile(f)]
    for file in files:
        ext = os.path.splitext(file)[1]
        for folder, extensii in foldere.items():
            if ext in extensii or file in extensii:
                try:
                    shutil.move(file, os.path.join(folder, file))
                    print(f"Mutat: {file} -> {folder}")
                except Exception as e:
                    print(f"Nu s-a putut muta {file}: {e}")

if __name__ == "__main__":
    organizeaza_proiectul()