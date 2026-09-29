#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Outil de prédiction hebdomadaire - Boulangerie (2 Olivaer Platz, Berlin)
==========================================================================

Entrée   : une date future correspondant à un LUNDI (ex : 2026-11-02)
Sortie   : un fichier .xlsx contenant, pour les 7 jours de la semaine
           (lundi -> dimanche) :
             - les ventes prédites par produit (viennoiseries suivies
               dans l'onglet "Data" du fichier source)
             - le chiffre d'affaires prédit (total boulangerie + détail
               par produit suivi)
             - la météo (prévision réelle ou, au-delà de l'horizon de
               prévision, une climatologie construite à partir des
               années précédentes)

Utilisation
-----------
    python predire_semaine.py --date 2026-11-02
    python predire_semaine.py --date 2026-11-02 --data "Data_Plaquage_Clean.xlsx" --output "predictions.xlsx"

Le script réentraîne les modèles à chaque exécution à partir du fichier
source (rapide : quelques secondes), il n'y a donc pas de fichier modèle
à maintenir séparément. Le fichier source doit rester à jour (les
nouvelles données de vente peuvent y être ajoutées au fil du temps pour
améliorer les prédictions futures).

Dépendances : pandas, numpy, scikit-learn, openpyxl, requests
    pip install pandas numpy scikit-learn openpyxl requests

API météo utilisée : Open-Meteo (https://open-meteo.com), gratuite et
sans clé d'API.
    - Prévision (<= 16 jours) : /v1/forecast
    - Climatologie (> 16 jours) : moyenne des 5 dernières années via
      /v1/archive
"""

import argparse
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

def predict_week(monday, model_a, model_b, prix_tiers, produits, weather_by_day):
    week_dates = [monday + timedelta(days=i) for i in range(7)]

    rows_produits = []
    rows_jour = []

    for d in week_dates:
        w = weather_by_day[d]
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
            rows_produits.append({
                "Date": d, "Jour": JOURS_FR[d.weekday()], "Produit": p,
                "Quantité prédite": int(q), "Prix unitaire (€)": prix_p,
                "CA produit prédit (€)": round(ca_p, 2),
            })

        # --- prédiction CA total boulangerie ---
        X_ca = pd.DataFrame([base])
        ca_total_pred = float(model_b.predict(X_ca)[0])
        ca_total_pred = max(ca_total_pred, 0)

        rows_jour.append({
            "Date": d, "Jour": JOURS_FR[d.weekday()],
            "Mois": MOIS_FR[d.month - 1],
            "Condition météo": w["Condition Principale"],
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
TITLE_FONT = Font(name="Arial", bold=True, size=14, color="2F5496")
NOTE_FONT = Font(name="Arial", italic=True, size=9, color="666666")
THIN = Side(style="thin", color="D9D9D9")
BORDER = Border(left=THIN, right=THIN, top=THIN, bottom=THIN)


def _style_header_row(ws, row_idx, n_cols):
    for c in range(1, n_cols + 1):
        cell = ws.cell(row=row_idx, column=c)
        cell.fill = HEADER_FILL
        cell.font = HEADER_FONT
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        cell.border = BORDER


def _write_df(ws, df, start_row=1, start_col=1):
    for j, col in enumerate(df.columns):
        ws.cell(row=start_row, column=start_col + j, value=col)
    _style_header_row(ws, start_row, len(df.columns))
    for i, (_, row) in enumerate(df.iterrows(), start=1):
        for j, col in enumerate(df.columns):
            val = row[col]
            if isinstance(val, (pd.Timestamp, datetime, date)):
                val = pd.Timestamp(val).to_pydatetime().date()
            cell = ws.cell(row=start_row + i, column=start_col + j, value=val)
            cell.font = BASE_FONT
            cell.border = BORDER
            if isinstance(val, date) and not isinstance(val, datetime):
                cell.number_format = "dd/mm/yyyy"
    for j, col in enumerate(df.columns):
        width = max(12, min(38, len(str(col)) + 4,
                             max((len(str(v)) for v in df[col]), default=8) + 4))
        ws.column_dimensions[get_column_letter(start_col + j)].width = width
    return start_row + len(df) + 1


def write_output(path, monday, df_jour, df_produits, metrics_a, metrics_b, source_note):
    wb = openpyxl.Workbook()

    # --- Feuille 1 : résumé semaine ---
    ws = wb.active
    ws.title = "Résumé semaine"
    ws["A1"] = f"Prédictions boulangerie - semaine du {monday.strftime('%d/%m/%Y')}"
    ws["A1"].font = TITLE_FONT
    ws["A2"] = "2 Olivaer Platz, Berlin"
    ws["A2"].font = NOTE_FONT
    next_row = _write_df(ws, df_jour, start_row=4)

    total_ca = df_jour["CA total boulangerie prédit (€)"].sum()
    ws.cell(row=next_row + 1, column=1, value="Total semaine - CA boulangerie prédit (€)").font = Font(name="Arial", bold=True)
    ws.cell(row=next_row + 1, column=2, value=round(total_ca, 2)).font = Font(name="Arial", bold=True)

    ws.cell(row=next_row + 3, column=1,
            value="Note : \"CA total boulangerie\" est estimé directement à partir de l'historique météo/CA "
                  "(toutes ventes). \"CA viennoiserie suivie\" ne couvre que les produits détaillés dans "
                  "l'onglet Data du fichier source.").font = NOTE_FONT
    ws.cell(row=next_row + 4, column=1, value=source_note).font = NOTE_FONT
    ws.cell(row=next_row + 5, column=1,
            value=f"Qualité modèle ventes par produit (validation) : MAE={metrics_a['MAE']:.1f} unités, "
                  f"R²={metrics_a['R2']:.2f}").font = NOTE_FONT
    ws.cell(row=next_row + 6, column=1,
            value=f"Qualité modèle CA total (validation) : MAE={metrics_b['MAE']:.0f} €, "
                  f"R²={metrics_b['R2']:.2f}").font = NOTE_FONT

    # --- Feuille 2 : détail ventes par produit ---
    ws2 = wb.create_sheet("Détail ventes produits")
    ws2["A1"] = "Détail des ventes prédites par produit et par jour"
    ws2["A1"].font = TITLE_FONT
    _write_df(ws2, df_produits, start_row=3)

    # --- Feuille 3 : tableau croisé produit x jour (quantités) ---
    ws3 = wb.create_sheet("Ventes (vue croisée)")
    pivot = df_produits.pivot_table(index="Produit", columns="Jour",
                                     values="Quantité prédite", aggfunc="sum")
    pivot = pivot.reindex(columns=[j for j in JOURS_FR if j in pivot.columns])
    pivot = pivot.reset_index()
    ws3["A1"] = "Quantités prédites par produit (vue croisée)"
    ws3["A1"].font = TITLE_FONT
    _write_df(ws3, pivot, start_row=3)

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
    parser = argparse.ArgumentParser(description="Prédiction hebdomadaire boulangerie (ventes, CA, météo).")
    parser.add_argument("--date", required=True, help="Lundi de la semaine à prédire, format AAAA-MM-JJ")
    parser.add_argument("--data", default="Data_Plaquage_Clean.xlsx", help="Chemin vers le fichier source")
    parser.add_argument("--output", default=None, help="Chemin du fichier de sortie .xlsx")
    args = parser.parse_args()

    monday = parse_monday(args.date)
    output_path = args.output or f"Predictions_Semaine_{monday.isoformat()}.xlsx"

    print(f"Chargement des données depuis {args.data} ...")
    df_ventes, df_meteo, prix_tiers = load_source_data(args.data)
    produits = sorted(df_ventes["Produit"].unique())
    print(f"{len(df_ventes)} lignes de vente, {len(df_meteo)} jours de météo, {len(produits)} produits suivis.")

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

    df_jour, df_produits = predict_week(monday, model_a, model_b, prix_tiers, produits, weather_by_day)

    write_output(output_path, monday, df_jour, df_produits, metrics_a, metrics_b, source_note)
    print(f"Fichier généré : {output_path}")


if __name__ == "__main__":
    main()
