import os
import pandas as pd

# Definire coloană de căutat
coloane_cautate = ['Tip_Masurare']
fisier_gasit_corect = False

# Caută toate fișierele .csv din folderul curent și din subfoldere
for root, dirs, files in os.walk('.'):
    for file in files:
        if file.endswith('.csv'):
            cale_completa = os.path.join(root, file)
            try:
                # se citește doar capul de tabel inițial
                df = pd.read_csv(cale_completa, nrows=5)

                coloana_gasita = None
                for col in coloane_cautate:
                    if col in df.columns:
                        coloana_gasita = col
                        break

                # Daca s-a găsit coloana in acest fisier
                if coloana_gasita:
                    fisier_gasit_corect = True
                    print(f"\nAm gasit coloana '{coloana_gasita}' in fisierul: {cale_completa}")

                    # Acum se citește tot fisierul pentru a extrage categoriile
                    df_complet = pd.read_csv(cale_completa)
                    categorii = sorted(df_complet[coloana_gasita].dropna().astype(str).unique())


                    for cod, nume in enumerate(categorii):
                        if cod == 0:
                            culoare = 'Albastru (Valoare minimă pe axa SHAP)'
                        elif cod == len(categorii) - 1:
                            culoare = 'Roșu (Valoare maximă pe axa SHAP)'
                        else:
                            culoare = 'Mov (Valoare intermediară)'

                        print(f" Cod {cod} [ Culoare SHAP: {culoare} ] ---> {nume}")

                    break  # Iese din bucla fisierelor
            except Exception:
                pass  # Ignoră fisierele care nu pot fi citite

    if fisier_gasit_corect:
        break  # Iese din bucla directoarelor

if not fisier_gasit_corect:
    print("\nNiciun fișier .csv nu contine coloana dorita.")