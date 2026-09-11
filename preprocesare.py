import pandas as pd
import os


# 1. Încărcare dicționar
try:
    dictionar = pd.read_csv('dictionar.csv')
    print("Dicționarul a fost încărcat.")
except FileNotFoundError:
    print("EROARE: Nu găsesc 'dictionar.csv'.")
    exit()

# Lista fișierelor excel
file_list = [
    'Masuratori dinamice interior.xlsx',
    'Masuratori statice exterior.xlsx',
    'Masuratori statice interior.xlsx'
]

# Aici se colectează datele
all_data_frames = []

# 2. Se proceseaza fiecare fișier Excel
for file_name in file_list:
    if not os.path.exists(file_name):
        print(f"Fișierul '{file_name}' nu a fost găsit.")
        continue

    print(f"Procesez fișierul: {file_name}...")

    # Se extrag caracteristicile din numele fișierului
    tip_masurare = "dinamica" if "dinamice" in file_name else "statica"
    locatie = "interior" if "interior" in file_name else "exterior"

    # Se mapeaza distanțele pentru interior vs. exterior
    if locatie == "interior":
        # Astea sunt numele coloanelor din excel-ul de interior
        dist_cols = ['Freq 0.3 Hz', 'Total 0.3 µT', 'Freq 0.9 Hz', 'Total 0.9 µT', 'Freq 1.5 Hz', 'Total 1.5 µT']
        # Astea sunt valorile reale ale distanțelor
        dist_values = [0.3, 0.9, 1.5]
    else:  # exterior
        # Astea sunt numele coloanelor din excel-ul de exterior
        dist_cols = ['Freq 0.5 Hz', 'Total 0.5 µT', 'Freq 1.5 Hz', 'Total 1.5 µT', 'Freq 2.5 Hz', 'Total 2.5 µT']
        dist_values = [0.5, 1.5, 2.5]

    # Se încarcă fișierul excel
    xls = pd.ExcelFile(file_name)

    # Se iterează prin fiecare sheet (Pct 1, Pct A, Ambiental, etc.)
    for sheet_name in xls.sheet_names:
        print(f"  ...citesc sheet-ul: {sheet_name}")
        df = pd.read_excel(xls, sheet_name=sheet_name)

        # Logica pentru Ambiental
        if sheet_name == 'Ambiental':
            try:
                df_ambient = df[['Freq Hz', 'Total µT']].copy()
                df_ambient = df_ambient.rename(columns={'Freq Hz': 'Frecventa', 'Total µT': 'Total_B'})
                df_ambient['Distanta'] = pd.NA  # Folosim NA (Not Available)
                df_ambient['Punct_Masurare'] = 'Ambiental'
                df_ambient['Tip_Masurare'] = tip_masurare
                df_ambient['Locatie'] = locatie
                all_data_frames.append(df_ambient)
            except KeyError as e:
                print(f"    EROARE la sheet-ul 'Ambiental': Nu găsesc coloana {e}.")

        # Logica pentru sheet-urile "Pct X"
        else:
            try:

                # Se procesează prima distanță (folosind index 0 și 1 din liste)
                df_dist1 = df[[dist_cols[0], dist_cols[1]]].copy()
                df_dist1 = df_dist1.rename(columns={dist_cols[0]: 'Frecventa', dist_cols[1]: 'Total_B'})
                df_dist1['Distanta'] = dist_values[0]

                # Se procesează a doua distanță (folosind index 2 și 3 din liste)
                df_dist2 = df[[dist_cols[2], dist_cols[3]]].copy()
                df_dist2 = df_dist2.rename(columns={dist_cols[2]: 'Frecventa', dist_cols[3]: 'Total_B'})
                df_dist2['Distanta'] = dist_values[1]

                # Se procesează a treia distanță (folosind index 4 și 5 din liste)
                df_dist3 = df[[dist_cols[4], dist_cols[5]]].copy()
                df_dist3 = df_dist3.rename(columns={dist_cols[4]: 'Frecventa', dist_cols[5]: 'Total_B'})
                df_dist3['Distanta'] = dist_values[2]

                # Se combină cele 3 seturi de date de la acest Punct
                df_punct = pd.concat([df_dist1, df_dist2, df_dist3])

                # Se adaugă caracteristicile
                df_punct['Punct_Masurare'] = sheet_name  # ex: "Pct 1"
                df_punct['Tip_Masurare'] = tip_masurare
                df_punct['Locatie'] = locatie

                all_data_frames.append(df_punct)

            except KeyError as e:
                print(f"    EROARE la sheet-ul '{sheet_name}': Nu găsesc coloana {e}.")
            except Exception as e:
                print(f"    EROARE necunoscută la '{sheet_name}': {e}")

# 3. Se combină toate datele
print("Combin datele...")
master_data = pd.concat(all_data_frames, ignore_index=True)

# 4. Se adaugă informațiile din dicționar
print("Adaug informațiile din dicționar...")
final_master_data = pd.merge(master_data,dictionar,on='Punct_Masurare',how='left')

# 5. Salvare
final_master_data.to_csv('Preprocesare.csv', index=False)

print("---")
print(f"Total rânduri preprocesate: {len(final_master_data)}")
print(final_master_data.head())
print(final_master_data.tail())