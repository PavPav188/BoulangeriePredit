#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Outil de prédiction hebdomadaire - Boulangerie (2 Olivaer Platz, Berlin)
==========================================================================

Entrée   : une date future correspondant à un LUNDI (ex : 2026-11-02)
Sortie   : un fichier .xlsx contenant :
             1. "Météo & CA"          : prévision météo (températures et
                                         conditions) + CA estimé pour chaque
                                         jour de lundi à dimanche
             2. "Plaquage & ventes"   : ventes prédites par produit et par
                                         jour + plaquage recommandé (en
                                         plaques ou en pièces), comparé au
                                         plaquage habituel
             3. "Détail ventes produits" : détail long (produit x jour, avec
                                         prix et CA par produit)

Utilisation
-----------
    python predire_semaine.py --date 2026-11-02
    python predire_semaine.py --date 2026-11-02 --data "Data_Plaquage_Clean.xlsx" --output "predictions.xlsx"
    python predire_semaine.py --date 2026-11-02 --marge 0.05

Règles de plaquage
------------------
- Croissants et petites cramiques : plaques de 12
- Pains au chocolat : plaques de 14
- Autres produits (grosses cramiques, navettes, sandwichs, croissants
  amande) : comptés à la pièce
- Le plaquage recommandé = ventes prédites (+ marge éventuelle), arrondies
  à la plaque supérieure.
- Si le modèle prédit 0 vente pour un produit que l'on plaque d'habitude
  (produits à faible volume), le plaquage habituel est repris.
- Le plaquage "habituel" sert de point de comparaison (voir
  PLAQUAGE_HABITUEL_SEMAINE / PLAQUAGE_HABITUEL_WEEKEND ci-dessous).
  Pour les produits non listés (petites cramiques, croissants amande), il
  est calculé à partir de la médiane historique du fichier source.

Dépendances : pandas, numpy, scikit-learn, openpyxl, requests
    pip install pandas numpy scikit-learn openpyxl requests

API météo utilisée : Open-Meteo (https://open-meteo.com), gratuite et
sans clé d'API.
    - Prévision (<= 16 jours) : /v1/forecast
    - Climatologie (> 16 jours) : moyenne des 5 dernières années via
      /v1/archive
"""

import argparse
import math
import re
import sys
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
import requests
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

# Localisation de la boulangerie : 2 Olivaer Platz, 10629 Berlin
LATITUDE = 52.5025
LONGITUDE = 13.3125
TIMEZONE = "Europe/Berlin"

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_HORIZON_DAYS = 16  # horizon fiable d'Open-Meteo

JOURS_FR = ["lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi", "dimanche"]
MOIS_FR = ["janvier", "février", "mars", "avril", "mai", "juin", "juillet",
           "août", "septembre", "octobre", "novembre", "décembre"]

# Traduction des conditions (format Visual Crossing du fichier source)
CONDITION_FR = {
    "Clear": "Dégagé",
    "Partially cloudy": "Partiellement nuageux",
    "Overcast": "Couvert",
    "Rain": "Pluie",
    "Snow": "Neige",
}

# Correspondance code météo OMM (Open-Meteo) -> catégories utilisées dans
# le fichier source (colonne "Condition Principale", format Visual Crossing)
WMO_TO_CONDITION = {
    0: "Clear", 1: "Clear",
    2: "Partially cloudy",
    3: "Overcast",
    45: "Overcast", 48: "Overcast",                 # brouillard
    51: "Rain", 53: "Rain", 55: "Rain",              # bruine
    56: "Rain", 57: "Rain",                          # bruine verglaçante
    61: "Rain", 63: "Rain", 65: "Rain",
    66: "Rain", 67: "Rain",
    71: "Snow", 73: "Snow", 75: "Snow", 77: "Snow",
    80: "Rain", 81: "Rain", 82: "Rain",
    85: "Snow", 86: "Snow",
    95: "Rain", 96: "Rain", 99: "Rain",
}

# --- Plaquage --------------------------------------------------------------

# Nombre de pièces par plaque. Tout produit absent = comptage à la pièce (1).
TAILLE_PLAQUE = {
    "Croissant": 12,
    "Pain chocolat": 14,
    "Pt Cr Nature": 12,
    "Pt Cr Sucre": 12,
    "Pt Cr Raisin": 12,
    "Pt Cr Chocolat": 12,
}

# Plaquage habituel, exprimé dans l'unité du produit :
#   nombre de PLAQUES si le produit est dans TAILLE_PLAQUE, sinon nombre de PIÈCES.
PLAQUAGE_HABITUEL_SEMAINE = {   # lundi -> vendredi
    "Croissant": 12,
    "Pain chocolat": 5,
    "Navette": 42,
    "Sandwich": 16,
    "Gr Cr Nature": 4,
    "Gr Cr Raisin": 4,
    "Gr Cr Choco": 4,
    "Gr Cr Sucre": 2,
    "Gr Cr R&S": 2,
}
PLAQUAGE_HABITUEL_WEEKEND = {   # samedi, dimanche
    "Croissant": 24,
    "Pain chocolat": 10,
    "Navette": 63,
    "Sandwich": 16,
    "Gr Cr Nature": 6,
    "Gr Cr Raisin": 6,
    "Gr Cr Choco": 6,
    "Gr Cr Sucre": 2,
    "Gr Cr R&S": 2,
}


def taille_plaque(produit):
    return TAILLE_PLAQUE.get(produit, 1)


def libelle_unite(produit):
    t = taille_plaque(produit)
    return f"plaques de {t}" if t > 1 else "pièces"


def wmo_to_condition(code):
    if code is None or (isinstance(code, float) and np.isnan(code)):
        return "Partially cloudy"
    return WMO_TO_CONDITION.get(int(code), "Partially cloudy")


# --------------------------------------------------------------------------
# 1. Chargement et préparation des données historiques
# --------------------------------------------------------------------------

def load_source_data(path):
    """Charge les onglets utiles du fichier source et construit :
       - df_ventes  : historique journalier des ventes par produit (long format)
       - df_meteo   : historique journalier météo + CA total boulangerie
       - prix_tiers : grille tarifaire datée, voir load_price_tiers()
    """
    xls = pd.ExcelFile(path)

    df_data = pd.read_excel(xls, sheet_name="Data")
    df_meteo = pd.read_excel(xls, sheet_name="Meteo")
    df_prix = pd.read_excel(xls, sheet_name="Data Prix")

    df_data = df_data.dropna(subset=["Date"]).copy()
    df_meteo = df_meteo.dropna(subset=["Date"]).copy()

    df_data["Date"] = pd.to_datetime(df_data["Date"]).dt.normalize()
    df_meteo["Date"] = pd.to_datetime(df_meteo["Date"]).dt.normalize()

    default_year = int(df_data["Date"].dt.year.max())
    prix_tiers = load_price_tiers(df_prix, default_year)

    # Ventes réelles = mis en vente (Plaquage) - invendu (Reste)
    df_data["Vendu"] = (df_data["Valeur Plaquage"].fillna(0)
                         - df_data["Valeur Reste"].fillna(0)).clip(lower=0)
    df_data = df_data.rename(columns={"Produit Plaquage": "Produit"})

    # Fusion avec la météo du jour (mêmes dates)
    meteo_cols = ["Date", "Température Moyenne", "Température Max",
                  "Température Min", "Condition Principale", "CA"]
    df_ventes = df_data.merge(df_meteo[meteo_cols], on="Date", how="inner")

    df_ventes = add_calendar_features(df_ventes)
    df_meteo = add_calendar_features(df_meteo)

    return df_ventes, df_meteo, prix_tiers


PRICE_COL_PATTERN = re.compile(r"apr[eè]s\s+(\d{1,2})/(\d{1,2})(?:/(\d{2,4}))?", re.IGNORECASE)


def load_price_tiers(df_prix, default_year):
    """Construit une grille tarifaire à partir de l'onglet "Data Prix".

    La colonne "Prix" (sans mention de date) sert de tarif de base.
    Toute colonne supplémentaire nommée "Prix après JJ/MM" ou
    "Prix après JJ/MM/AAAA" définit un nouveau tarif applicable
    STRICTEMENT APRÈS cette date (l'année est déduite de l'historique
    du fichier si elle n'est pas précisée dans l'en-tête).

    Renvoie une liste [(date_effet_ou_None, {produit: prix}), ...] triée
    chronologiquement, `None` désignant le tarif de base (toujours valable
    en premier lieu).
    """
    cols = list(df_prix.columns)
    base_col = None
    dated = []  # (date_effet, nom_colonne)

    for col in cols[1:]:
        m = PRICE_COL_PATTERN.search(str(col))
        if m:
            day, month, year = m.groups()
            year = int(year) if year else default_year
            dated.append((date(year, int(month), int(day)), col))
        elif base_col is None:
            base_col = col

    if base_col is None:
        raise ValueError("Aucune colonne de prix de base trouvée dans l'onglet 'Data Prix'.")

    tiers = [(None, dict(zip(df_prix[cols[0]], df_prix[base_col])))]
    for eff_date, col in sorted(dated, key=lambda t: t[0]):
        tiers.append((eff_date, dict(zip(df_prix[cols[0]], df_prix[col]))))
    return tiers


def get_price(prix_tiers, d, produit):
    """Prix applicable pour `produit` à la date `d`, selon la grille
    tarifaire (dernier tarif dont la date d'effet est strictement
    antérieure à `d`)."""
    price = prix_tiers[0][1].get(produit, 0)
    for eff_date, prices in prix_tiers[1:]:
        if d > eff_date:
            price = prices.get(produit, price)
        else:
            break
    return price


def add_calendar_features(df):
    df = df.copy()
    df["jour_num"] = df["Date"].dt.weekday  # lundi = 0
    df["mois_num"] = df["Date"].dt.month
    df["jour_annee"] = df["Date"].dt.dayofyear
    return df


def build_plaquage_habituel(df_ventes, produits):
    """Construit {(produit, est_weekend): plaquage habituel}, exprimé dans
    l'unité du produit (plaques ou pièces).

    Les valeurs saisies dans PLAQUAGE_HABITUEL_* sont prioritaires. Pour les
    autres produits (petites cramiques, croissants amande...), on prend la
    médiane historique du plaquage, arrondie à la plaque la plus proche.
    """
    habituel = {}
    weekend_flag = df_ventes["jour_num"] >= 5
    for p in produits:
        t = taille_plaque(p)
        for est_we, table in ((False, PLAQUAGE_HABITUEL_SEMAINE),
                              (True, PLAQUAGE_HABITUEL_WEEKEND)):
            if p in table:
                habituel[(p, est_we)] = table[p]
                continue
            sub = df_ventes[(df_ventes["Produit"] == p) & (weekend_flag == est_we)]
            med = float(sub["Valeur Plaquage"].median()) if len(sub) else 0.0
            habituel[(p, est_we)] = int(round(med / t))
    return habituel


# --------------------------------------------------------------------------
# 2. Entraînement des modèles
# --------------------------------------------------------------------------

NUM_FEATURES = ["Température Moyenne", "Température Max", "Température Min",
                 "jour_num", "mois_num", "jour_annee"]


def train_product_model(df_ventes):
    """Modèle A : prédit la quantité vendue, pour un produit et un jour
    donnés, à partir du calendrier et de la météo."""
    cat_features = ["Produit", "Condition Principale"]
    X = df_ventes[NUM_FEATURES + cat_features]
    y = df_ventes["Vendu"]

    preprocess = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features),
    ], remainder="passthrough")

    model = Pipeline([
        ("prep", preprocess),
        ("rf", RandomForestRegressor(n_estimators=300, max_depth=None,
                                      min_samples_leaf=2, random_state=42,
                                      n_jobs=-1)),
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    metrics = {
        "MAE": mean_absolute_error(y_test, pred),
        "R2": r2_score(y_test, pred),
    }

    # Ré-entraînement final sur 100% des données pour la prédiction réelle
    model.fit(X, y)
    return model, metrics


def train_ca_model(df_meteo):
    """Modèle B : prédit le CA TOTAL de la boulangerie (toutes ventes, pas
    seulement les produits suivis dans l'onglet Data) à partir du calendrier
    et de la météo du jour."""
    cat_features = ["Condition Principale"]
    X = df_meteo[NUM_FEATURES + cat_features]
    y = df_meteo["CA"]

    preprocess = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features),
    ], remainder="passthrough")

    model = Pipeline([
        ("prep", preprocess),
        ("rf", RandomForestRegressor(n_estimators=300, max_depth=None,
                                      min_samples_leaf=2, random_state=42,
                                      n_jobs=-1)),
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    metrics = {
        "MAE": mean_absolute_error(y_test, pred),
        "R2": r2_score(y_test, pred),
    }

    model.fit(X, y)
    return model, metrics


# --------------------------------------------------------------------------
# 3. Météo future (prévision Open-Meteo ou climatologie de repli)
# --------------------------------------------------------------------------

def fetch_forecast(start_date, end_date):
    """Prévision réelle Open-Meteo (fiable jusqu'à ~16 jours)."""
    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "daily": "weathercode,temperature_2m_max,temperature_2m_min,temperature_2m_mean",
        "timezone": TIMEZONE,
    }
    r = requests.get(FORECAST_URL, params=params, timeout=20)
    r.raise_for_status()
    d = r.json()["daily"]

    out = {}
    for i, day_str in enumerate(d["time"]):
        out[datetime.strptime(day_str, "%Y-%m-%d").date()] = {
            "Température Moyenne": d["temperature_2m_mean"][i],
            "Température Max": d["temperature_2m_max"][i],
            "Température Min": d["temperature_2m_min"][i],
            "Condition Principale": wmo_to_condition(d["weathercode"][i]),
            "Source météo": "Prévision (Open-Meteo)",
        }
    return out


def fetch_climatology(week_dates, years_back=5):
    """Au-delà de l'horizon de prévision : moyenne des conditions
    observées les mêmes jours du calendrier au cours des `years_back`
    dernières années (archive Open-Meteo)."""
    per_position = {i: {"tmoy": [], "tmax": [], "tmin": [], "codes": []}
                     for i in range(len(week_dates))}

    this_year = week_dates[0].year
    for back in range(1, years_back + 1):
        year = this_year - back
        try:
            start = week_dates[0].replace(year=year)
            end = week_dates[-1].replace(year=year)
        except ValueError:
            # 29 février sans équivalent
            continue

        params = {
            "latitude": LATITUDE,
            "longitude": LONGITUDE,
            "start_date": start.isoformat(),
            "end_date": end.isoformat(),
            "daily": "weathercode,temperature_2m_max,temperature_2m_min,temperature_2m_mean",
            "timezone": TIMEZONE,
        }
        try:
            r = requests.get(ARCHIVE_URL, params=params, timeout=20)
            r.raise_for_status()
            d = r.json()["daily"]
        except Exception:
            continue

        for i in range(min(len(d["time"]), len(week_dates))):
            if d["temperature_2m_mean"][i] is None:
                continue
            per_position[i]["tmoy"].append(d["temperature_2m_mean"][i])
            per_position[i]["tmax"].append(d["temperature_2m_max"][i])
            per_position[i]["tmin"].append(d["temperature_2m_min"][i])
            per_position[i]["codes"].append(d["weathercode"][i])

    out = {}
    for i, day in enumerate(week_dates):
        vals = per_position[i]
        if not vals["tmoy"]:
            # repli ultime : normales saisonnières approximatives (Berlin)
            out[day] = {
                "Température Moyenne": 10.0, "Température Max": 14.0,
                "Température Min": 6.0, "Condition Principale": "Partially cloudy",
                "Source météo": "Climatologie (valeur par défaut, données archive indisponibles)",
            }
            continue
        codes = pd.Series(vals["codes"])
        mode_code = codes.mode().iloc[0] if not codes.mode().empty else codes.iloc[0]
        out[day] = {
            "Température Moyenne": round(float(np.mean(vals["tmoy"])), 1),
            "Température Max": round(float(np.mean(vals["tmax"])), 1),
            "Température Min": round(float(np.mean(vals["tmin"])), 1),
            "Condition Principale": wmo_to_condition(mode_code),
            "Source météo": f"Climatologie (moyenne sur {len(vals['tmoy'])} année(s) passée(s))",
        }
    return out


def get_week_weather(week_dates):
    """Renvoie {date: {...}} pour les 7 jours, en utilisant la prévision
    Open-Meteo si l'horizon le permet, sinon une climatologie."""
    today = date.today()
    if week_dates[-1] <= today + timedelta(days=FORECAST_HORIZON_DAYS):
        try:
            return fetch_forecast(week_dates[0], week_dates[-1])
        except Exception as e:
            print(f"[Attention] Échec de la prévision Open-Meteo ({e}), "
                  f"bascule sur la climatologie.", file=sys.stderr)
    return fetch_climatology(week_dates)


# --------------------------------------------------------------------------
# 4. Prédiction pour la semaine demandée
# --------------------------------------------------------------------------

def predict_week(monday, model_a, model_b, prix_tiers, produits, weather_by_day,
                 habituel, marge=0.0):
    """Renvoie trois DataFrames :
       - df_jour     : météo + CA estimé par jour
       - df_produits : détail produit x jour (ventes, prix, CA, plaquage)
    """
    week_dates = [monday + timedelta(days=i) for i in range(7)]

    rows_produits = []
    rows_jour = []

    for d in week_dates:
        w = weather_by_day[d]
        est_we = d.weekday() >= 5
        base = {
            "Température Moyenne": w["Température Moyenne"],
            "Température Max": w["Température Max"],
            "Température Min": w["Température Min"],
            "jour_num": d.weekday(),
            "mois_num": d.month,
            "jour_annee": d.timetuple().tm_yday,
            "Condition Principale": w["Condition Principale"],
        }

        # --- prédiction par produit ---
        X_prod = pd.DataFrame([{**base, "Produit": p} for p in produits])
        qte_pred = model_a.predict(X_prod)
        qte_pred = np.clip(np.round(qte_pred), 0, None)

        ca_produits_du_jour = 0.0
        for p, q in zip(produits, qte_pred):
            prix_p = get_price(prix_tiers, d, p)
            ca_p = q * prix_p
            ca_produits_du_jour += ca_p

            # --- plaquage recommandé : ventes (+ marge) arrondies à la plaque sup.
            t = taille_plaque(p)
            pieces_reco = int(math.ceil(q * (1 + marge) / t - 1e-9)) * t
            unites_reco = pieces_reco // t            # plaques (ou pièces si t = 1)
            unites_hab = habituel.get((p, est_we), 0)
            # Produits à faible volume (ex : Pt Cr Sucre) : le modèle peut prédire 0
            # alors qu'on en plaque toujours un peu -> on retombe sur l'habituel.
            if unites_reco == 0 and unites_hab > 0:
                unites_reco = unites_hab
                pieces_reco = unites_reco * t

            rows_produits.append({
                "Date": d, "Jour": JOURS_FR[d.weekday()], "Produit": p,
                "Quantité prédite": int(q), "Prix unitaire (€)": prix_p,
                "CA produit prédit (€)": round(ca_p, 2),
                "Unité de plaquage": libelle_unite(p),
                "Plaquage recommandé (unités)": unites_reco,
                "Plaquage recommandé (pièces)": pieces_reco,
                "Plaquage habituel (unités)": unites_hab,
                "Écart vs habituel (unités)": unites_reco - unites_hab,
                "Invendu attendu (pièces)": max(pieces_reco - int(q), 0),
            })

        # --- prédiction CA total boulangerie ---
        X_ca = pd.DataFrame([base])
        ca_total_pred = float(model_b.predict(X_ca)[0])
        ca_total_pred = max(ca_total_pred, 0)

        rows_jour.append({
            "Date": d, "Jour": JOURS_FR[d.weekday()],
            "Mois": MOIS_FR[d.month - 1],
            "Condition météo": CONDITION_FR.get(w["Condition Principale"],
                                                w["Condition Principale"]),
            "Temp. min (°C)": w["Température Min"],
            "Temp. moy (°C)": w["Température Moyenne"],
            "Temp. max (°C)": w["Température Max"],
            "Source météo": w["Source météo"],
            "Ventes viennoiserie (unités)": int(qte_pred.sum()),
            "CA viennoiserie suivie (€)": round(ca_produits_du_jour, 2),
            "CA total boulangerie prédit (€)": round(ca_total_pred, 2),
        })

    return pd.DataFrame(rows_jour), pd.DataFrame(rows_produits)


# --------------------------------------------------------------------------
# 5. Écriture du fichier Excel de sortie
# --------------------------------------------------------------------------

HEADER_FILL = PatternFill("solid", fgColor="2F5496")
HEADER_FONT = Font(name="Arial", bold=True, color="FFFFFF", size=11)
BASE_FONT = Font(name="Arial", size=10)
BOLD_FONT = Font(name="Arial", size=10, bold=True)
TITLE_FONT = Font(name="Arial", bold=True, size=14, color="2F5496")
BLOCK_FONT = Font(name="Arial", bold=True, size=11, color="2F5496")
NOTE_FONT = Font(name="Arial", italic=True, size=9, color="666666")
WEEKEND_FILL = PatternFill("solid", fgColor="FFF2CC")
TOTAL_FILL = PatternFill("solid", fgColor="D9E2F3")
THIN = Side(style="thin", color="D9D9D9")
BORDER = Border(left=THIN, right=THIN, top=THIN, bottom=THIN)


def _style_header_row(ws, row_idx, n_cols, start_col=1):
    for c in range(start_col, start_col + n_cols):
        cell = ws.cell(row=row_idx, column=c)
        cell.fill = HEADER_FILL
        cell.font = HEADER_FONT
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        cell.border = BORDER


def _write_df(ws, df, start_row=1, start_col=1, autosize=True):
    for j, col in enumerate(df.columns):
        ws.cell(row=start_row, column=start_col + j, value=col)
    _style_header_row(ws, start_row, len(df.columns), start_col)
    for i, (_, row) in enumerate(df.iterrows(), start=1):
        for j, col in enumerate(df.columns):
            val = row[col]
            if isinstance(val, (np.integer,)):
                val = int(val)
            elif isinstance(val, (np.floating,)):
                val = float(val)
            if isinstance(val, (pd.Timestamp, datetime, date)):
                val = pd.Timestamp(val).to_pydatetime().date()
            cell = ws.cell(row=start_row + i, column=start_col + j, value=val)
            cell.font = BASE_FONT
            cell.border = BORDER
            if isinstance(val, date) and not isinstance(val, datetime):
                cell.number_format = "dd/mm/yyyy"
    if autosize:
        for j, col in enumerate(df.columns):
            width = max(12, min(38, len(str(col)) + 4,
                                 max((len(str(v)) for v in df[col]), default=8) + 4))
            ws.column_dimensions[get_column_letter(start_col + j)].width = width
    return start_row + len(df) + 1


def _fmt_col(ws, col_idx, first_row, last_row, number_format):
    for r in range(first_row, last_row + 1):
        ws.cell(row=r, column=col_idx).number_format = number_format


def _pivot_block(df_produits, produits, jours_labels, value_col, unit_col=True):
    """Tableau produit x jour pour une colonne de valeurs donnée."""
    pv = df_produits.pivot_table(index="Produit", columns="Jour_label",
                                  values=value_col, aggfunc="sum")
    pv = pv.reindex(index=produits, columns=jours_labels)
    pv = pv.reset_index()
    if unit_col:
        unites = df_produits.drop_duplicates("Produit").set_index("Produit")["Unité de plaquage"]
        pv.insert(1, "Unité", pv["Produit"].map(unites))
    return pv


def write_output(path, monday, df_jour, df_produits, metrics_a, metrics_b,
                 source_note, marge):
    wb = openpyxl.Workbook()
    dates = [monday + timedelta(days=i) for i in range(7)]
    jours_labels = [f"{JOURS_FR[d.weekday()]} {d.strftime('%d/%m')}" for d in dates]

    # ======================================================================
    # Feuille 1 : Météo & CA
    # ======================================================================
    ws = wb.active
    ws.title = "Météo & CA"
    ws["A1"] = f"Météo et CA estimé - semaine du {monday.strftime('%d/%m/%Y')}"
    ws["A1"].font = TITLE_FONT
    ws["A2"] = "2 Olivaer Platz, Berlin"
    ws["A2"].font = NOTE_FONT

    df_meteo_ca = pd.DataFrame({
        "Date": df_jour["Date"],
        "Jour": df_jour["Jour"],
        "Condition": df_jour["Condition météo"],
        "Temp. min (°C)": df_jour["Temp. min (°C)"],
        "Temp. moy (°C)": df_jour["Temp. moy (°C)"],
        "Temp. max (°C)": df_jour["Temp. max (°C)"],
        "CA estimé (€)": df_jour["CA total boulangerie prédit (€)"],
        "Source météo": df_jour["Source météo"],
    })
    header_row = 4
    next_row = _write_df(ws, df_meteo_ca, start_row=header_row)
    first, last = header_row + 1, header_row + len(df_meteo_ca)
    _fmt_col(ws, 7, first, last, '#,##0 "€"')
    for r in range(first, last + 1):
        if dates[r - first].weekday() >= 5:
            for c in range(1, 9):
                ws.cell(row=r, column=c).fill = WEEKEND_FILL

    total_row = last + 1
    ws.cell(row=total_row, column=1, value="Total semaine")
    ws.cell(row=total_row, column=7, value=f"=SUM(G{first}:G{last})")
    ws.cell(row=total_row, column=7).number_format = '#,##0 "€"'
    for c in range(1, 9):
        cell = ws.cell(row=total_row, column=c)
        cell.font = BOLD_FONT
        cell.fill = TOTAL_FILL
        cell.border = BORDER

    ws.cell(row=total_row + 2, column=1,
            value="CA estimé = CA total de la boulangerie (toutes ventes), modèle entraîné sur "
                  "l'historique météo/CA du fichier source.").font = NOTE_FONT
    ws.cell(row=total_row + 3, column=1, value=source_note).font = NOTE_FONT
    ws.cell(row=total_row + 4, column=1,
            value=f"Qualité du modèle CA (validation) : MAE = {metrics_b['MAE']:.0f} €, "
                  f"R² = {metrics_b['R2']:.2f}").font = NOTE_FONT
    ws.column_dimensions["C"].width = 24
    ws.column_dimensions["H"].width = 44
    ws.freeze_panes = "A5"

    # ======================================================================
    # Feuille 2 : Plaquage & ventes
    # ======================================================================
    ws2 = wb.create_sheet("Plaquage & ventes")
    ws2["A1"] = f"Plaquage à prévoir et ventes estimées - semaine du {monday.strftime('%d/%m/%Y')}"
    ws2["A1"].font = TITLE_FONT
    ws2["A2"] = ("Croissants et petites cramiques : plaques de 12 - Pains au chocolat : plaques de 14 - "
                 "Autres produits : à la pièce. Plaquage recommandé = ventes prédites arrondies à la "
                 "plaque supérieure" + (f" (marge de {marge:.0%})." if marge else "."))
    ws2["A2"].font = NOTE_FONT

    dfp = df_produits.copy()
    dfp["Jour_label"] = [f"{j} {pd.Timestamp(d).strftime('%d/%m')}"
                          for j, d in zip(dfp["Jour"], dfp["Date"])]
    produits = list(dict.fromkeys(dfp["Produit"]))

    blocks = [
        ("1. Ventes estimées (pièces)", "Quantité prédite", True, "sum"),
        ("2. Plaquage recommandé (nombre de plaques, ou de pièces pour les produits non plaqués)",
         "Plaquage recommandé (unités)", True, None),
        ("3. Plaquage habituel (même unité)", "Plaquage habituel (unités)", True, None),
        ("4. Écart recommandé - habituel (même unité)", "Écart vs habituel (unités)", True, None),
        ("5. Plaquage recommandé en pièces", "Plaquage recommandé (pièces)", True, "sum"),
        ("6. Invendu attendu (pièces)", "Invendu attendu (pièces)", True, "sum"),
    ]

    row = 4
    for title, col, unit_col, total in blocks:
        ws2.cell(row=row, column=1, value=title).font = BLOCK_FONT
        row += 1
        pv = _pivot_block(dfp, produits, jours_labels, col, unit_col=unit_col)
        hdr = row
        row = _write_df(ws2, pv, start_row=row, autosize=False)
        first_b, last_b = hdr + 1, hdr + len(pv)

        # colonnes week-end en jaune (samedi, dimanche = 2 dernières colonnes)
        n_cols = len(pv.columns)
        for r in range(first_b, last_b + 1):
            for c in (n_cols - 1, n_cols):
                ws2.cell(row=r, column=c).fill = WEEKEND_FILL
            for c in range(3, n_cols + 1):
                ws2.cell(row=r, column=c).alignment = Alignment(horizontal="center")

        # écart : mise en forme +/-
        if col == "Écart vs habituel (unités)":
            for r in range(first_b, last_b + 1):
                for c in range(3, n_cols + 1):
                    ws2.cell(row=r, column=c).number_format = '+0;-0;0'

        # ligne de total (uniquement en pièces, pour rester cohérent entre produits)
        if total == "sum":
            ws2.cell(row=row, column=1, value="Total")
            for c in range(3, n_cols + 1):
                L = get_column_letter(c)
                ws2.cell(row=row, column=c, value=f"=SUM({L}{first_b}:{L}{last_b})")
            for c in range(1, n_cols + 1):
                cell = ws2.cell(row=row, column=c)
                cell.font = BOLD_FONT
                cell.fill = TOTAL_FILL
                cell.border = BORDER
                if c >= 3:
                    cell.alignment = Alignment(horizontal="center")
            row += 1
        row += 2

    ws2.cell(row=row, column=1,
             value=f"Qualité du modèle ventes par produit (validation) : "
                   f"MAE = {metrics_a['MAE']:.1f} unités, R² = {metrics_a['R2']:.2f}").font = NOTE_FONT
    ws2.cell(row=row + 1, column=1,
             value="Plaquage habituel : valeurs saisies dans le script pour croissants, pains au chocolat, "
                   "navettes, sandwichs et grosses cramiques ; médiane historique pour les petites "
                   "cramiques et croissants amande.").font = NOTE_FONT
    ws2.column_dimensions["A"].width = 20
    ws2.column_dimensions["B"].width = 16
    for c in range(3, 10):
        ws2.column_dimensions[get_column_letter(c)].width = 15
    ws2.freeze_panes = "C4"

    # ======================================================================
    # Feuille 3 : Détail ventes produits (annexe)
    # ======================================================================
    ws3 = wb.create_sheet("Détail ventes produits")
    ws3["A1"] = "Détail des ventes prédites par produit et par jour"
    ws3["A1"].font = TITLE_FONT
    cols_detail = ["Date", "Jour", "Produit", "Quantité prédite", "Prix unitaire (€)",
                   "CA produit prédit (€)", "Unité de plaquage",
                   "Plaquage recommandé (unités)", "Plaquage recommandé (pièces)"]
    _write_df(ws3, df_produits[cols_detail], start_row=3)

    wb.save(path)


# --------------------------------------------------------------------------
# 6. Point d'entrée
# --------------------------------------------------------------------------

def parse_monday(date_str):
    try:
        d = datetime.strptime(date_str, "%Y-%m-%d").date()
    except ValueError:
        raise SystemExit(f"Format de date invalide : '{date_str}'. Utilisez AAAA-MM-JJ (ex : 2026-11-02).")
    if d.weekday() != 0:
        raise SystemExit(f"La date {d.strftime('%d/%m/%Y')} n'est pas un lundi "
                          f"(c'est un {JOURS_FR[d.weekday()]}). Merci de fournir un lundi.")
    return d


def main():
    parser = argparse.ArgumentParser(description="Prédiction hebdomadaire boulangerie (ventes, CA, météo, plaquage).")
    parser.add_argument("--date", required=True, help="Lundi de la semaine à prédire, format AAAA-MM-JJ")
    parser.add_argument("--data", default="Data_Plaquage_Clean.xlsx", help="Chemin vers le fichier source")
    parser.add_argument("--output", default=None, help="Chemin du fichier de sortie .xlsx")
    parser.add_argument("--marge", type=float, default=0.0,
                        help="Marge de sécurité appliquée aux ventes prédites avant arrondi "
                             "à la plaque (ex : 0.05 = +5%%). Défaut : 0")
    args = parser.parse_args()

    monday = parse_monday(args.date)
    output_path = args.output or f"Predictions_Semaine_{monday.isoformat()}.xlsx"

    print(f"Chargement des données depuis {args.data} ...")
    df_ventes, df_meteo, prix_tiers = load_source_data(args.data)
    produits = sorted(df_ventes["Produit"].unique())
    print(f"{len(df_ventes)} lignes de vente, {len(df_meteo)} jours de météo, {len(produits)} produits suivis.")
    habituel = build_plaquage_habituel(df_ventes, produits)

    print("Entraînement du modèle de ventes par produit ...")
    model_a, metrics_a = train_product_model(df_ventes)
    print(f"  -> MAE={metrics_a['MAE']:.1f} unités, R²={metrics_a['R2']:.2f}")

    print("Entraînement du modèle de CA total boulangerie ...")
    model_b, metrics_b = train_ca_model(df_meteo)
    print(f"  -> MAE={metrics_b['MAE']:.0f} €, R²={metrics_b['R2']:.2f}")

    week_dates = [monday + timedelta(days=i) for i in range(7)]
    print(f"Récupération de la météo pour la semaine du {monday} ...")
    weather_by_day = get_week_weather(week_dates)
    source_note = f"Météo : {next(iter(weather_by_day.values()))['Source météo']}"
    print(f"  -> {source_note}")

    df_jour, df_produits = predict_week(monday, model_a, model_b, prix_tiers, produits,
                                        weather_by_day, habituel, marge=args.marge)

    write_output(output_path, monday, df_jour, df_produits, metrics_a, metrics_b,
                 source_note, args.marge)
    print(f"Fichier généré : {output_path}")


if __name__ == "__main__":
    main()
