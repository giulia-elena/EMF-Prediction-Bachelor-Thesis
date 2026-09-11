
import customtkinter as ctk
import joblib
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Import pentru ANN
try:
    from tensorflow.keras.models import load_model
except ImportError:
    from keras.models import load_model


# DEFINIȚIA CLASEI - pentru modelul Hybrid
class SafetyFirstPredictor:
    def __init__(self, model_baza, safety_factor=3.0):
        self.model = model_baza
        self.safety_factor = safety_factor
        self.mae_istoric = 0.001673

    def predict_safe(self, X):
        pred_base = self.model.predict(X)
        if hasattr(pred_base, 'flatten'): pred_base = pred_base.flatten()
        bias = self.safety_factor * self.mae_istoric
        val_base = max(0, float(pred_base[0]))
        pred_safe = val_base + bias
        return val_base, [pred_safe]


# -------------------------------------------------------------
# SETĂRI TEMĂ: Light Mode implicit
# -------------------------------------------------------------
ctk.set_appearance_mode("light")
ctk.set_default_color_theme("blue")


class SimulatorSSM_Pro(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Sistem de Analiză Multi-Model și Conformitate Legală EMC")
        self.geometry("1400x850")
        # Setăm un tuplu: (Culoare_Light, Culoare_Dark)
        self.configure(fg_color=("#e2e8f0", "#121212"))

        self.encoders, self.scaler, self.rf_model, self.ann_model, self.hybrid_model = [None] * 5
        self.incarca_resurse_ai()

        # 1. BAZA DE DATE COMPLETĂ
        self.mapping = {
            "dinamica": {
                "interior": {
                    "Pct A": {"sursa": "Aparat aer condiționat mecanic", "zona": "655-5-A Interior",
                              "dist": ["0.9", "1.5"]},
                    "Pct B": {"sursa": "Boghiu motor", "zona": "655-5-A Interior", "dist": ["0.3", "0.9", "1.5"]},
                    "Pct C": {"sursa": "Aer condiționat călători", "zona": "655-5-A Interior",
                              "dist": ["0.3", "0.9", "1.5"]},
                    "Pct D": {"sursa": "Boghiu motor", "zona": "655-5-A Interior", "dist": ["0.3", "0.9", "1.5"]},
                    "Pct E": {"sursa": "Transformator auxiliar", "zona": "655-5-C Interior",
                              "dist": ["0.3", "0.9", "1.5"]},
                    "Pct F": {"sursa": "Aer condiționat călători", "zona": "655-5-C Interior",
                              "dist": ["0.3", "0.9", "1.5"]},
                    "Pct G": {"sursa": "Transformator auxiliar", "zona": "655-5-C Interior",
                              "dist": ["0.3", "0.9", "1.5"]},
                    "Pct H": {"sursa": "Pantograf", "zona": "655-5-B Interior", "dist": ["0.3", "0.9", "1.5"]},
                    "Pct I": {"sursa": "Transformator de tracțiune", "zona": "655-5-B Interior",
                              "dist": ["0.3", "0.9", "1.5"]},
                    "Pct J": {"sursa": "Aer condiționat călători", "zona": "655-5-B Interior",
                              "dist": ["0.3", "0.9", "1.5"]},
                    "Pct K": {"sursa": "Pantograf", "zona": "655-5-B Interior", "dist": ["0.3", "0.9", "1.5"]},
                    "Pct L": {"sursa": "Aparat aer condiționat mecanic", "zona": "655-5-A Interior",
                              "dist": ["0.9", "1.5"]},
                    "Ambiental": {"sursa": "Fundal", "zona": "Ambiental Interior", "dist": ["0.0"]}
                }
            },
            "statica": {
                "interior": {
                    "Pct A": {"sursa": "Aparat aer condiționat mecanic", "zona": "655-5-A Interior",
                              "dist": ["0.9", "1.5"]},
                    "Pct B": {"sursa": "Boghiu motor", "zona": "655-5-A Interior", "dist": ["0.3", "0.9", "1.5"]},
                    "Pct C_F_J": {"sursa": "Aer condiționat călători", "zona": "Zone 655-5-A+B+C Interior",
                                  "dist": ["0.3", "0.9", "1.5"]},
                    "Pct D": {"sursa": "Boghiu motor", "zona": "655-5-A Interior", "dist": ["0.3", "0.9", "1.5"]},
                    "Pct E_G": {"sursa": "Transformator auxiliar", "zona": "Zone 655-5-C Interior",
                                "dist": ["0.3", "0.9", "1.5"]},
                    "Pct H_K": {"sursa": "Pantograf", "zona": "655-5-B Interior", "dist": ["0.3", "0.9", "1.5"]},
                    "Pct I": {"sursa": "Transformator de tracțiune", "zona": "655-5-B Interior",
                              "dist": ["0.3", "0.9", "1.5"]},
                    "Pct L": {"sursa": "Aparat aer condiționat mecanic", "zona": "655-5-A Interior",
                              "dist": ["0.9", "1.5"]},
                    "Ambiental": {"sursa": "Fundal", "zona": "Ambiental Interior", "dist": ["0.0"]}
                },
                "exterior": {
                    "Pct 1": {"sursa": "Aparat aer condiționat mecanic", "zona": "655-5-A Exterior",
                              "dist": ["0.5", "1.5", "2.5"]},
                    "Pct 2": {"sursa": "Boghiu motor stânga/dreapta", "zona": "655-5-A Exterior",
                              "dist": ["0.5", "1.5", "2.5"]},
                    "Pct 3": {"sursa": "Boghiu motor stânga/dreapta", "zona": "655-5-A Exterior",
                              "dist": ["0.5", "1.5", "2.5"]},
                    "Pct 4": {"sursa": "Aer condiționat călători stânga/dreapta", "zona": "655-5-A Exterior",
                              "dist": ["0.5", "1.5", "2.5"]},
                    "Pct 5": {"sursa": "Aer condiționat călători stânga/dreapta", "zona": "655-5-A Exterior",
                              "dist": ["0.5", "1.5", "2.5"]},
                    "Pct 6": {"sursa": "Boghiu motor stânga/dreapta", "zona": "655-5-A Exterior",
                              "dist": ["0.5", "1.5", "2.5"]},
                    "Pct 7": {"sursa": "Boghiu motor stânga/dreapta", "zona": "655-5-A Exterior",
                              "dist": ["0.5", "1.5", "2.5"]},
                    "Pct 8": {"sursa": "Transformator auxiliar stânga/dreapta", "zona": "655-5-C Exterior",
                              "dist": ["0.5", "1.5", "2.5"]},
                    "Pct 9": {"sursa": "Transformator auxiliar stânga/dreapta", "zona": "655-5-C Exterior",
                              "dist": ["0.5", "1.5", "2.5"]},
                    "Pct 10_18": {"sursa": "Aer condiționat călători stânga/dreapta", "zona": "Zone 655-5-B+C Exterior",
                                  "dist": ["0.5", "1.5", "2.5"]},
                    "Pct 11": {"sursa": "Aer condiționat călători stânga/dreapta", "zona": "655-5-C Exterior",
                               "dist": ["0.5", "1.5", "2.5"]},
                    "Pct 12": {"sursa": "Transformator auxiliar stânga/dreapta", "zona": "655-5-C Exterior",
                               "dist": ["0.5", "1.5", "2.5"]},
                    "Pct 13": {"sursa": "Transformator auxiliar stânga/dreapta", "zona": "655-5-C Exterior",
                               "dist": ["0.5", "1.5", "2.5"]},
                    "Pct 14_21": {"sursa": "Pantograf stânga/dreapta", "zona": "655-5-B Exterior",
                                  "dist": ["0.5", "1.5", "2.5"]},
                    "Pct 15": {"sursa": "Fundal", "zona": "655-5-B Exterior", "dist": ["0.5", "1.5", "2.5"]},
                    "Pct 16": {"sursa": "Transformator tracțiune stânga/dreapta", "zona": "655-5-B Exterior",
                               "dist": ["0.5", "1.5", "2.5"]},
                    "Pct 17": {"sursa": "Transformator tracțiune stânga/dreapta", "zona": "655-5-B Exterior",
                               "dist": ["0.5", "1.5", "2.5"]},
                    "Pct 19": {"sursa": "Aer condiționat călători stânga/dreapta", "zona": "655-5-B Exterior",
                               "dist": ["0.5", "1.5", "2.5"]},
                    "Pct 20": {"sursa": "Pantograf stânga/dreapta", "zona": "655-5-B Exterior",
                               "dist": ["0.5", "1.5", "2.5"]},
                    "Pct 22": {"sursa": "Aparat aer condiționat mecanic", "zona": "655-5-A Exterior",
                               "dist": ["0.5", "1.5", "2.5"]},
                    "Ambiental": {"sursa": "Fundal", "zona": "Ambiental Exterior", "dist": ["0.0"]}
                }
            }
        }

        # --- CONSTRUCȚIE INTERFAȚĂ ---
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # =====================================================================
        # BARA LATERALĂ
        # =====================================================================
        self.sidebar = ctk.CTkScrollableFrame(self, width=420, corner_radius=0, fg_color=("#f1f5f9", "#1e1e1e"))
        self.sidebar.grid(row=0, column=0, sticky="nsew")

        ctk.CTkLabel(self.sidebar, text="PANOU DE CONTROL", font=ctk.CTkFont(size=18, weight="bold"),
                     text_color=("#334155", "#e2e8f0")).pack(pady=(15, 5))

        # TRENULEȚUL MODERN
        self.frame_vehicul = ctk.CTkFrame(self.sidebar, fg_color=("#dbeafe", "#1e3a8a"), corner_radius=10,
                                          border_width=1,
                                          border_color=("#bfdbfe", "#1e40af"))
        self.frame_vehicul.pack(pady=5, padx=20, fill="x")
        ctk.CTkLabel(self.frame_vehicul, text="🚄", font=ctk.CTkFont(size=40)).pack(pady=(5, 0))
        ctk.CTkLabel(self.frame_vehicul, text="RAMĂ ELECTRICĂ", font=ctk.CTkFont(weight="bold", size=15),
                     text_color=("#1e40af", "#93c5fd")).pack(pady=(0, 2))
        ctk.CTkLabel(self.frame_vehicul, text="25 kV | 160 km/h | 4x105 kW", font=ctk.CTkFont(size=11),
                     text_color=("#3b82f6", "#60a5fa")).pack(pady=(0, 10))

        # CONTROALE INPUT
        font_label = ctk.CTkFont(size=12, weight="bold")
        color_label = ("#475569", "#cbd5e1")
        input_bg = ("#ffffff", "#2b2b2b")
        text_primary = ("#0f172a", "#f8f9fa")
        btn_color = ("#e2e8f0", "#404040")
        btn_hover = ("#cbd5e1", "#475569")

        ctk.CTkLabel(self.sidebar, text="1. Regim de funcționare:", text_color=color_label, font=font_label).pack(
            padx=20, anchor="w", pady=(10, 0))
        self.opt_regim = ctk.CTkOptionMenu(self.sidebar, values=["Dinamic", "Static"],
                                           command=self.update_locatie_options, fg_color=input_bg,
                                           text_color=text_primary,
                                           button_color=btn_color, button_hover_color=btn_hover)
        self.opt_regim.pack(pady=(2, 5), padx=20, fill="x")

        ctk.CTkLabel(self.sidebar, text="2. Locația măsurătorii:", text_color=color_label, font=font_label).pack(
            padx=20, anchor="w")
        self.opt_loc = ctk.CTkOptionMenu(self.sidebar, values=[], command=self.update_puncte_options, fg_color=input_bg,
                                         text_color=text_primary, button_color=btn_color, button_hover_color=btn_hover)
        self.opt_loc.pack(pady=(2, 5), padx=20, fill="x")

        ctk.CTkLabel(self.sidebar, text="3. Punct (ID | Zonă | Sursă):", text_color=color_label, font=font_label).pack(
            padx=20, anchor="w")
        self.opt_punct = ctk.CTkOptionMenu(self.sidebar, values=[], command=self.on_punct_select, fg_color=input_bg,
                                           text_color=text_primary, button_color=btn_color,
                                           button_hover_color=btn_hover)
        self.opt_punct.pack(pady=(2, 5), padx=20, fill="x")

        ctk.CTkLabel(self.sidebar, text="4. Distanța față de sursă (m):", text_color=color_label, font=font_label).pack(
            padx=20, anchor="w")
        self.opt_dist = ctk.CTkOptionMenu(self.sidebar, values=[], fg_color=input_bg, text_color=text_primary,
                                          button_color=btn_color, button_hover_color=btn_hover)
        self.opt_dist.pack(pady=(2, 5), padx=20, fill="x")

        ctk.CTkLabel(self.sidebar, text="5. Frecvența de lucru (Hz):", text_color=color_label, font=font_label).pack(
            padx=20, anchor="w")
        self.ent_freq = ctk.CTkEntry(self.sidebar, placeholder_text="Ex: 50.0", fg_color=input_bg,
                                     text_color=text_primary)
        self.ent_freq.pack(pady=(2, 2), padx=20, fill="x")
        self.ent_freq.bind("<KeyRelease>", self.valideaza_frecventa)

        self.lbl_freq_status = ctk.CTkLabel(self.sidebar, text="Interval permis: 4.5 - 20025 Hz",
                                            font=ctk.CTkFont(size=11), text_color=("#94a3b8", "#64748b"))
        self.lbl_freq_status.pack(padx=20, anchor="w", pady=(0, 5))

        # DEMO ALERTE
        ctk.CTkLabel(self.sidebar, text="MOD DEMO (TESTARE ALERTE)", text_color=("#ea580c", "#f97316"),
                     font=font_label).pack(
            padx=20, anchor="w")
        self.opt_demo = ctk.CTkOptionMenu(self.sidebar,
                                          values=["Normal (Real)", "Atenție (Portocaliu)", "Pericol (Roșu)"],
                                          fg_color=("#fff7ed", "#431407"), text_color=("#c2410c", "#fdba74"),
                                          button_color=("#ffedd5", "#7c2d12"),
                                          button_hover_color=("#fed7aa", "#9a3412"))
        self.opt_demo.pack(pady=(2, 10), padx=20, fill="x")

        # SELECTOR MODELE AI CU SCORURI
        ctk.CTkLabel(self.sidebar, text="6. Selectați Modelele AI:", font=font_label, text_color=color_label).pack(
            pady=(5, 2), padx=20, anchor="w")
        self.model_vars = {
            "Random Forest (Stabil | R²=0.9452)": ctk.BooleanVar(value=True),
            "Neural Network (Extrapolare | R²=0.8942)": ctk.BooleanVar(value=True),
            "Safety-First (RF + Marjă 3xMAE)": ctk.BooleanVar(value=True)
        }
        for name, var in self.model_vars.items():
            ctk.CTkCheckBox(self.sidebar, text=name, variable=var, text_color=("#334155", "#e2e8f0"),
                            border_color=("#94a3b8", "#64748b"), hover_color="#2563eb", fg_color="#2563eb").pack(pady=2,
                                                                                                                 padx=30,
                                                                                                                 anchor="w")

        # Buton Principal
        self.btn_calc = ctk.CTkButton(self.sidebar, text="ANALIZEAZĂ CONFORMITATEA", height=50,
                                      font=ctk.CTkFont(size=15, weight="bold"), fg_color="#2563eb",
                                      hover_color="#1d4ed8")
        self.btn_calc.configure(command=self.simulare_calcul)
        self.btn_calc.pack(pady=(15, 10), padx=20, fill="x")

        # COMUTATOR DARK MODE NOU ADAUGAT AICI
        self.switch_theme = ctk.CTkSwitch(self.sidebar, text="Dark Mode", command=self.schimba_tema,
                                          font=ctk.CTkFont(size=12, weight="bold"), text_color=color_label)
        self.switch_theme.pack(pady=(0, 20), padx=20, anchor="w")

        # =====================================================================
        # --- PANOU CENTRAL (DASHBOARD) ---
        # =====================================================================
        self.main_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.main_frame.grid(row=0, column=1, padx=30, pady=15, sticky="nsew")

        ctk.CTkLabel(self.main_frame, text="Monitorizare și Diagnoză EMC", font=ctk.CTkFont(size=26, weight="bold"),
                     text_color=text_primary).pack(pady=(0, 10), anchor="w")

        # Setari culori comune pentru carduri
        card_bg = ("#ffffff", "#242424")
        card_border = ("#cbd5e1", "#404040")
        card_title = ("#64748b", "#94a3b8")

        # WIDGET 1: REZULTATUL PRINCIPAL
        self.card_rezultat = ctk.CTkFrame(self.main_frame, corner_radius=12, fg_color=card_bg, border_width=1,
                                          border_color=card_border)
        self.card_rezultat.pack(pady=(0, 15), fill="x")

        ctk.CTkLabel(self.card_rezultat, text="🛡️ PREDICȚIE MAXIMĂ DE EXPUNERE",
                     font=ctk.CTkFont(size=13, weight="bold"), text_color=card_title).pack(pady=(10, 0))
        self.res_display = ctk.CTkLabel(self.card_rezultat, text="--.----- µT",
                                        font=ctk.CTkFont(size=50, weight="bold"), text_color="#10b981")
        self.res_display.pack(pady=(0, 0))
        self.lbl_status_text = ctk.CTkLabel(self.card_rezultat, text="Așteptare date intrare...",
                                            font=ctk.CTkFont(size=13, weight="bold"), text_color=card_title)
        self.lbl_status_text.pack(pady=(0, 10))

        # WIDGET 2: GRAFICUL VIZUAL
        self.card_grafic = ctk.CTkFrame(self.main_frame, corner_radius=12, fg_color=card_bg, border_width=1,
                                        border_color=card_border)
        self.card_grafic.pack(pady=(0, 15), fill="x")

        ctk.CTkLabel(self.card_grafic, text="📊 ANALIZĂ VIZUALĂ A CONFORMITĂȚII",
                     font=ctk.CTkFont(size=13, weight="bold"), text_color=card_title).pack(pady=(10, 0), anchor="w",
                                                                                           padx=20)
        self.graph_frame = ctk.CTkFrame(self.card_grafic, fg_color="transparent")
        self.graph_frame.pack(pady=5, padx=15, fill="both")
        self.canvas_widget = None

        # =====================================================================
        # WIDGET 3: ZONA DE JOS (Raport + Indicator)
        # =====================================================================
        self.bottom_frame = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        self.bottom_frame.pack(pady=0, fill="both", expand=True)

        self.bottom_frame.grid_columnconfigure(0, weight=2)
        self.bottom_frame.grid_columnconfigure(1, weight=1)
        self.bottom_frame.grid_rowconfigure(0, weight=1)

        # 3A: RAPORT DIAGNOZĂ (Stânga)
        self.card_raport = ctk.CTkFrame(self.bottom_frame, corner_radius=12, fg_color=card_bg, border_width=1,
                                        border_color=card_border)
        self.card_raport.grid(row=0, column=0, padx=(0, 10), sticky="nsew")

        ctk.CTkLabel(self.card_raport, text="📝 JURNAL DE DIAGNOZĂ TEHNICĂ", font=ctk.CTkFont(size=13, weight="bold"),
                     text_color=card_title).pack(pady=(10, 0), anchor="w", padx=20)

        # Folosim fontul Consolas (monospațiat) pentru aliniere de tip "tabel" perfectă
        self.txt_diag = ctk.CTkTextbox(self.card_raport, font=ctk.CTkFont(family="Consolas", size=13),
                                       fg_color=card_bg, text_color=text_primary, border_width=0, wrap="word")
        self.txt_diag.pack(pady=(5, 10), padx=20, fill="both", expand=True)

        # 3B: WIDGET DINAMIC (Dreapta) - ICONIȚĂ ȘI BARĂ DE PROGRES
        self.card_widget = ctk.CTkFrame(self.bottom_frame, corner_radius=12, fg_color=card_bg, border_width=1,
                                        border_color=card_border)
        self.card_widget.grid(row=0, column=1, padx=(10, 0), sticky="nsew")

        ctk.CTkLabel(self.card_widget, text="🎛️ INDICATOR SIGURANȚĂ", font=ctk.CTkFont(size=13, weight="bold"),
                     text_color=card_title).pack(side="top", pady=(10, 0))

        self.lbl_procent = ctk.CTkLabel(self.card_widget, text="0% din limita ICNIRP",
                                        font=ctk.CTkFont(size=12, weight="bold"), text_color=card_title)
        self.lbl_procent.pack(side="bottom", pady=(0, 15))

        self.progress_bar = ctk.CTkProgressBar(self.card_widget, width=200, height=12, corner_radius=10,
                                               fg_color=("#e2e8f0", "#404040"))
        self.progress_bar.pack(side="bottom", pady=(0, 10))
        self.progress_bar.set(0)

        # Rama invizibilă din centru absoarbe tot spațiul, iar iconița e plasată fix la relx=0.5
        self.icon_container = ctk.CTkFrame(self.card_widget, fg_color="transparent")
        self.icon_container.pack(expand=True, fill="both")

        self.lbl_icon = ctk.CTkLabel(self.icon_container, text="⚪", font=ctk.CTkFont(size=80))
        self.lbl_icon.place(relx=0.5, rely=0.5, anchor="center")

        # Inițializare liste în cascadă la pornire
        self.opt_regim.set("Dinamic")
        self.update_locatie_options("Dinamic")

    # --- FUNCTIE NOUA PENTRU SCHIMBAREA TEMEI ---
    def schimba_tema(self):
        if self.switch_theme.get() == 1:
            ctk.set_appearance_mode("dark")
        else:
            ctk.set_appearance_mode("light")

        # Re-generam graficul pentru a i se actualiza culorile (doar daca s-a apasat deja pe Analizeaza)
        if self.res_display.cget("text") != "--.----- µT":
            self.simulare_calcul()

    # --- FUNCȚII LOGICĂ INTERFAȚĂ ---
    def valideaza_frecventa(self, event=None):
        val = self.ent_freq.get()
        try:
            f = float(val)
            if 4.5 <= f <= 20025:
                self.ent_freq.configure(border_color="#10b981")
                self.lbl_freq_status.configure(text="✅ Frecvență Validă", text_color="#10b981")
            else:
                self.ent_freq.configure(border_color="#f59e0b")
                self.lbl_freq_status.configure(text="⚠️ În afara intervalului antrenat", text_color="#f59e0b")
        except:
            if val == "":
                self.ent_freq.configure(border_color="#d1d5db")
                self.lbl_freq_status.configure(text="Interval permis: 4.5 - 20025 Hz",
                                               text_color=("#94a3b8", "#64748b"))
            else:
                self.ent_freq.configure(border_color="#ef4444")
                self.lbl_freq_status.configure(text="❌ Introduceți o valoare numerică", text_color="#ef4444")

    def incarca_resurse_ai(self):
        path = '03_Modele_Salvate'
        try:
            self.encoders = joblib.load(os.path.join(path, 'encoders.joblib'))
            self.scaler = joblib.load(os.path.join(path, 'scaler_ann_keras.joblib'))
            self.rf_model = joblib.load(os.path.join(path, 'model_RF.joblib'))
            self.ann_model = load_model(os.path.join(path, 'model_ann_keras.h5'), compile=False)
            self.hybrid_model = joblib.load(os.path.join(path, 'model_algoritm_custom_Safety_First.joblib'))
            print("Resurse AI încărcate cu succes.")
        except Exception as e:
            print(f"Eroare încărcare modele: {e}")

    def update_locatie_options(self, r):
        self.opt_loc.configure(values=["Interior"] if r == "Dinamic" else ["Interior", "Exterior"])
        self.opt_loc.set("Interior")
        self.update_puncte_options("Interior")

    def update_puncte_options(self, l):
        rk = "dinamica" if self.opt_regim.get() == "Dinamic" else "statica"
        pm = self.mapping[rk][l.lower()]
        pd = [f"{pid} | {info['zona']} | {info['sursa']}" for pid, info in pm.items()]
        self.opt_punct.configure(values=pd)
        self.opt_punct.set(pd[0])
        self.on_punct_select(pd[0])

    def on_punct_select(self, pt):
        rk = "dinamica" if self.opt_regim.get() == "Dinamic" else "statica"
        lk = self.opt_loc.get().lower()
        pid = pt.split(" | ")[0]
        ds = self.mapping[rk][lk][pid]['dist']
        self.opt_dist.configure(values=ds)
        self.opt_dist.set(ds[0])

    def calculeaza_limite_exacte(self, f):
        icnirp = 40000 / (f ** 2) if f < 8 else 5000 / f if f < 800 else 6.25
        eu_dir = 200000 / (
                f ** 2) if f < 8 else 25000 / f if f < 25 else 1000 if f < 300 else 300000 / f if f < 3000 else 100
        return icnirp, eu_dir

    # --- CALCUL ȘI PREDICȚIE ---
    def simulare_calcul(self):
        try:
            f = float(self.ent_freq.get())
            d = float(self.opt_dist.get())
            rk_label = "dinamica" if self.opt_regim.get() == "Dinamic" else "statica"
            lk_label = self.opt_loc.get().lower()
            pt_text = self.opt_punct.get()
            pid = pt_text.split(" | ")[0]
            info = self.mapping[rk_label][lk_label][pid]

            lim_i, lim_e = self.calculeaza_limite_exacte(f)
            predictii = {}
            demo = self.opt_demo.get()

            if self.encoders is None:
                raise ValueError("Modelele AI nu s-au încărcat corect din folderul '03_Modele_Salvate'.")

            if pid == "Ambiental":
                val_base = 0.00045
                if self.model_vars["Random Forest (Stabil | R²=0.9452)"].get(): predictii["Random Forest"] = val_base
                if self.model_vars["Neural Network (Extrapolare | R²=0.8942)"].get(): predictii[
                    "Neural Network"] = val_base
                if self.model_vars["Safety-First (RF + Marjă 3xMAE)"].get(): predictii[
                    "Safety-First Hybrid"] = val_base + 0.005019
            else:
                s_enc = self.encoders['sursa'].transform([info['sursa']])[0]
                z_enc = self.encoders['zona'].transform([info['zona']])[0]
                l_enc = self.encoders['locatie'].transform([lk_label])[0]
                t_enc = self.encoders['tip'].transform([rk_label])[0]

                X = pd.DataFrame([[f, d, s_enc, z_enc, l_enc, t_enc]],
                                 columns=['Frecventa', 'Distanta', 'Sursa_Encoded', 'Zona_Encoded', 'Locatie_Encoded',
                                          'Tip_Encoded'])
                X_scaled = self.scaler.transform(X.values)

                p_rf = max(0, float(self.rf_model.predict(X)[0]))
                p_ann = max(0, float(self.ann_model.predict(X_scaled, verbose=0).flatten()[0]))
                _, p_hy_list = self.hybrid_model.predict_safe(X)
                p_hy = max(0, float(p_hy_list[0]))

                factor = 1.0
                if "Portocaliu" in demo:
                    factor = (lim_i * 1.5) / max(p_hy, 0.001)
                elif "Roșu" in demo:
                    factor = (lim_e * 1.1) / max(p_hy, 0.001)

                if self.model_vars["Random Forest (Stabil | R²=0.9452)"].get(): predictii[
                    "Random Forest"] = p_rf * factor
                if self.model_vars["Neural Network (Extrapolare | R²=0.8942)"].get(): predictii[
                    "Neural Network"] = p_ann * factor
                if self.model_vars["Safety-First (RF + Marjă 3xMAE)"].get(): predictii[
                    "Safety-First Hybrid"] = p_hy * factor

            if not predictii:
                self.txt_diag.insert("0.0", "EROARE: Selectați cel puțin un model AI pentru analiză!\n")
                return

            model_critic = max(predictii, key=predictii.get)
            val_max = predictii[model_critic]

            procent_fata_de_limita_publica = (val_max / lim_i) * 100
            procent_afisare_bara = min(procent_fata_de_limita_publica, 100) / 100.0

            if val_max >= lim_e:
                sc = "#ef4444"
                status_txt = "STATUS: PERICOL (Depășire limită profesională)"
                iconita = "☢️"
            elif val_max >= lim_i:
                sc = "#f59e0b"
                status_txt = "STATUS: ATENȚIE (Depășire limită public)"
                iconita = "⚠️"
            else:
                sc = "#10b981"
                status_txt = "STATUS: CONFORM (Zonă sigură)"
                iconita = "🛡️"

            self.res_display.configure(text=f"{val_max:.5f} µT", text_color=sc)
            self.lbl_status_text.configure(text=status_txt, text_color=sc)

            self.lbl_icon.configure(text=iconita)
            self.progress_bar.configure(progress_color=sc)
            self.progress_bar.set(procent_afisare_bara)
            self.lbl_procent.configure(text=f"{procent_fata_de_limita_publica:.2f}% din limita publică")

            self.txt_diag.delete("0.0", "end")

            # self.txt_diag.insert("end", " 📋 REZUMAT DIAGNOZĂ TEHNICĂ\n")
            # self.txt_diag.insert("end", " ==================================================================\n\n")

            if pid == "Ambiental":
                self.txt_diag.insert("end", " ℹ️ INFO: ANALIZĂ NIVEL DE FUNDAL (Zgomot Magnetic)\n\n")

            self.txt_diag.insert("end", " 1. CONFIGURAȚIE SCENARIU:\n")
            self.txt_diag.insert("end", f"    > {'Regim funcționare:'.ljust(24)} {self.opt_regim.get().upper()}\n")
            self.txt_diag.insert("end", f"    > {'Amplasament:'.ljust(24)} {lk_label.upper()}\n")
            self.txt_diag.insert("end", f"    > {'Punct analizat:'.ljust(24)} {pid} ({info['zona']})\n")
            self.txt_diag.insert("end", f"    > {'Sursă emisie:'.ljust(24)} {info['sursa']}\n")
            self.txt_diag.insert("end", f"    > {'Parametri fizici:'.ljust(24)} {d} m distanță | {f} Hz frecvență\n\n")

            self.txt_diag.insert("end", " 2. ANALIZĂ PREDICTIVĂ MODELE AI:\n")
            for m, v in predictii.items():
                maxim_tag = " [VALOARE MAXIMĂ PREZISĂ]" if m == model_critic else ""
                self.txt_diag.insert("end", f"    > {(m + ':').ljust(24)} {v:.6f} µT{maxim_tag}\n")

            self.txt_diag.insert("end", "\n 3. EVALUARE CONFORMITATE LEGALĂ:\n")
            self.txt_diag.insert("end", f"    > {'Limită ICNIRP (Public):'.ljust(24)} {lim_i:.2f} µT\n")
            self.txt_diag.insert("end", f"    > {'Limită Dir. 2013/35:'.ljust(24)} {lim_e:.2f} µT\n")
            self.txt_diag.insert("end", f"    > {'Grad ocupare ICNIRP:'.ljust(24)} {(val_max / lim_i) * 100:.4f}%\n")

            self.deseneaza_grafic(val_max, lim_i, lim_e, sc)

        except Exception as e:
            self.txt_diag.delete("0.0", "end")
            self.txt_diag.insert("0.0",
                                 f"EROARE SISTEM: Asigurați-vă că ați introdus o frecvență validă.\nDetalii tehnice: {e}")

    # --- FUNCȚIE GENERARE GRAFIC MATPLOTLIB MODIFICATA PENTRU DARK MODE ---
    def deseneaza_grafic(self, predictie, lim_icnirp, lim_eu, culoare_bar):
        if self.canvas_widget:
            self.canvas_widget.get_tk_widget().destroy()

        plt.style.use('default')
        fig, ax = plt.subplots(figsize=(8, 2.2), dpi=100)

        # Verificăm ce temă este activă pentru a schimba culorile Matplotlib
        tema_activa = ctk.get_appearance_mode()
        bg_color = '#ffffff' if tema_activa == "Light" else '#242424'
        text_color = '#64748b' if tema_activa == "Light" else '#94a3b8'
        grid_color = '#f1f5f9' if tema_activa == "Light" else '#404040'

        fig.patch.set_facecolor(bg_color)
        ax.set_facecolor(bg_color)

        bars = ax.barh([''], [predictie], color=culoare_bar, height=0.2)

        ax.axvline(x=lim_icnirp, color='#f59e0b', linestyle='--', linewidth=1.5,
                   label=f'ICNIRP (Public): {lim_icnirp:.2f} µT')
        ax.axvline(x=lim_eu, color='#ef4444', linestyle='-.', linewidth=1.5,
                   label=f'Dir. 2013/35 (Prof): {lim_eu:.2f} µT')

        ax.set_xscale('log')
        ax.set_xlabel('Inducție Magnetică B [µT] (Scară Logaritmică)', fontsize=10, color=text_color, labelpad=10)

        ax.grid(axis='x', color=grid_color, linestyle='-', linewidth=1)
        ax.set_axisbelow(True)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_color(grid_color)
        ax.tick_params(axis='x', colors=text_color, labelsize=9)
        ax.tick_params(axis='y', left=False)

        ax.legend(loc='lower right', facecolor=bg_color, edgecolor=grid_color, labelcolor=text_color, fontsize=9)
        plt.tight_layout()

        canvas = FigureCanvasTkAgg(fig, master=self.graph_frame)
        canvas.draw()
        self.canvas_widget = canvas
        self.canvas_widget.get_tk_widget().pack(fill="both", expand=True)


if __name__ == "__main__":
    print("Se încarcă interfața și modelele (TensorFlow)... Așteptați câteva secunde.")
    app = SimulatorSSM_Pro()
    app.mainloop()