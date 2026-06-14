"""
TPT for Flights - FastAPI Backend
Havayolu bazli model entegrasyonu: Pegasus / AJet / THY
Her havayolu icin ayri XGBoost + RF modeli, en iyi model secilir.
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from pathlib import Path
from datetime import date, timedelta
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
import joblib
import json

# ─── Paths ───────────────────────────────────────────────────────────────────
BASE         = Path(__file__).resolve().parent
PROJECT_ROOT = BASE.parent.parent
AIRLINE_DIR  = PROJECT_ROOT / "models" / "airline"
FEATURES_CSV = PROJECT_ROOT / "data" / "processed" / "flights_features.csv"
WEATHER_CSV  = PROJECT_ROOT / "data" / "processed" / "weather_data.csv"

templates = Jinja2Templates(directory=str(BASE / "templates"))
app = FastAPI(title="TPT for Flights")

# ─── Constants ───────────────────────────────────────────────────────────────
BAYRAM_START = date(2026, 5, 27)
BAYRAM_END   = date(2026, 5, 30)

AIRLINES = ["Pegasus", "AJet", "THY"]

# Her havayolu icin en iyi algo (CV MAE'ye gore)
BEST_ALGO = {
    "Pegasus": "xgb",
    "AJet":    "rf",
    "THY":     "rf",
}

ROUTES = {
    "IST_AYT": "İstanbul → Antalya",
    "AYT_IST": "Antalya → İstanbul",
    "IST_ADB": "İstanbul → İzmir",
    "ADB_IST": "İzmir → İstanbul",
    "SAW_ESB": "İstanbul (SAW) → Ankara",
    "ESB_SAW": "Ankara → İstanbul (SAW)",
}

ROUTE_DEST_CITY = {
    "IST_AYT": "AYT", "AYT_IST": "IST",
    "IST_ADB": "ADB", "ADB_IST": "IST",
    "SAW_ESB": "ESB", "ESB_SAW": "IST",
}

CITY_NAMES = {"IST": "İstanbul", "AYT": "Antalya", "ESB": "Ankara", "ADB": "İzmir"}

# Rota bazli global min istatistikler (egitim verisinden)
ROUTE_STATS = {
    "IST_AYT": {"global_avg": 2427.09, "global_std": 983.65,  "avg_flights": 42, "avg_airlines": 3},
    "AYT_IST": {"global_avg": 2215.22, "global_std": 854.41,  "avg_flights": 37, "avg_airlines": 3},
    "IST_ADB": {"global_avg": 1767.60, "global_std": 610.21,  "avg_flights": 34, "avg_airlines": 3},
    "ADB_IST": {"global_avg": 2008.87, "global_std": 711.12,  "avg_flights": 36, "avg_airlines": 3},
    "SAW_ESB": {"global_avg": 1589.22, "global_std": 378.34,  "avg_flights": 17, "avg_airlines": 2},
    "ESB_SAW": {"global_avg": 1527.89, "global_std": 254.94,  "avg_flights": 12, "avg_airlines": 2},
}

WEEKDAY_TR = ["Pazartesi", "Salı", "Çarşamba", "Perşembe", "Cuma", "Cumartesi", "Pazar"]

HOUR_MULTIPLIERS = {
    0: 0.82, 1: 0.80, 2: 0.79, 3: 0.79, 4: 0.80, 5: 0.83,
    6: 0.90, 7: 0.97, 8: 1.05, 9: 1.10, 10: 1.08, 11: 1.06,
    12: 1.04, 13: 1.02, 14: 1.00, 15: 0.99, 16: 1.01, 17: 1.07,
    18: 1.12, 19: 1.10, 20: 1.06, 21: 1.00, 22: 0.93, 23: 0.87,
}

# ─── Startup: modelleri yukle ─────────────────────────────────────────────────
airline_models: dict = {}       # {"Pegasus": model, ...}
feature_cols:   dict = {}       # {"Pegasus": [...], ...}
hist_df:    pd.DataFrame = pd.DataFrame()
weather_df: pd.DataFrame = pd.DataFrame()


@app.on_event("startup")
def startup():
    global airline_models, feature_cols, hist_df, weather_df

    # Havayolu modellerini yukle
    fc_path = AIRLINE_DIR / "feature_columns.json"
    if fc_path.exists():
        feature_cols = json.loads(fc_path.read_text(encoding="utf-8"))

    for airline in AIRLINES:
        algo = BEST_ALGO[airline]
        name = airline.lower()
        try:
            if algo == "xgb":
                m = XGBRegressor()
                m.load_model(str(AIRLINE_DIR / f"{name}_xgb_model.json"))
            else:
                m = joblib.load(AIRLINE_DIR / f"{name}_rf_model.joblib")
            airline_models[airline] = m
        except Exception as e:
            print(f"[WARN] {airline} modeli yuklenemedi: {e}")

    if FEATURES_CSV.exists():
        hist_df = pd.read_csv(FEATURES_CSV)
        hist_df["depart_date"] = pd.to_datetime(hist_df["depart_date"])
        hist_df["route"] = hist_df["origin"] + "_" + hist_df["destination"]

    if WEATHER_CSV.exists():
        weather_df = pd.read_csv(WEATHER_CSV)
        weather_df["date"] = pd.to_datetime(weather_df["date"])

    print(f"[OK] Yuklenen modeller: {list(airline_models.keys())}")


# ─── Helpers ─────────────────────────────────────────────────────────────────
def classify_bayram(d: date) -> str:
    if BAYRAM_START <= d <= BAYRAM_END:
        return "Bayram"
    if (BAYRAM_START - timedelta(days=7)) <= d < BAYRAM_START:
        return "Bayram Oncesi"
    if BAYRAM_END < d <= (BAYRAM_END + timedelta(days=7)):
        return "Bayram Sonrasi"
    return "Normal"


def get_weather(city_code: str, target_date: date):
    if weather_df.empty:
        return None
    m = weather_df[
        (weather_df["city_code"] == city_code) &
        (weather_df["date"].dt.date == target_date)
    ]
    if m.empty:
        return None
    r = m.iloc[0]
    return {
        "temp_max": float(r["temp_max"]), "temp_min": float(r["temp_min"]),
        "precipitation": float(r["precipitation"]), "wind_speed": float(r["wind_speed"]),
        "condition": str(r["weather_condition"]), "is_bad": int(r["is_bad_weather"]),
    }


def build_airline_features(airline: str, route_code: str, depart_date: date) -> pd.DataFrame:
    """Havayolu modeli icin feature vektoru uret."""
    stats    = ROUTE_STATS[route_code]
    dest     = ROUTE_DEST_CITY[route_code]
    w        = get_weather(dest, depart_date)
    today    = date(2026, 4, 30)

    dtf      = (depart_date - today).days
    weekday  = depart_date.weekday()
    dom      = depart_date.day
    wom      = (dom - 1) // 7 + 1
    month    = depart_date.month
    is_bay   = 1 if BAYRAM_START <= depart_date <= BAYRAM_END else 0
    dtb      = ((BAYRAM_START - depart_date).days if depart_date < BAYRAM_START
                else (depart_date - BAYRAM_END).days if depart_date > BAYRAM_END else 0)
    prox     = float(np.exp(-dtb / 7))
    period   = classify_bayram(depart_date)

    temp_max = w["temp_max"]  if w else 22.0
    temp_min = w["temp_min"]  if w else 12.0
    precip   = w["precipitation"] if w else 0.0
    wind     = w["wind_speed"]    if w else 10.0
    cond     = w["condition"]     if w else "Acik"
    is_bad   = w["is_bad"]        if w else 0

    # Bayram period normalization (egitimde _ kullaniyor)
    period_norm = period.replace(" ", "_")

    feat = {
        "flight_count":     stats["avg_flights"],
        "days_to_flight":   dtf,
        "depart_weekday":   weekday,
        "is_weekend":       1 if weekday in [5, 6] else 0,
        "day_of_month":     dom,
        "week_of_month":    wom,
        "month":            month,
        "is_bayram":        is_bay,
        "days_to_bayram":   dtb,
        "bayram_proximity": prox,
        "is_red_eye_avail": 0,
        "route_global_avg": stats["global_avg"],
        "route_global_std": stats["global_std"],
        "temp_max":         temp_max,
        "temp_min":         temp_min,
        "precipitation":    precip,
        "wind_speed":       wind,
        "is_bad_weather":   is_bad,
        # Route one-hot
        "route_ADB_IST": 1 if route_code == "ADB_IST" else 0,
        "route_AYT_IST": 1 if route_code == "AYT_IST" else 0,
        "route_ESB_SAW": 1 if route_code == "ESB_SAW" else 0,
        "route_IST_ADB": 1 if route_code == "IST_ADB" else 0,
        "route_IST_AYT": 1 if route_code == "IST_AYT" else 0,
        "route_SAW_ESB": 1 if route_code == "SAW_ESB" else 0,
        # Bayram period one-hot
        "bayram_period_Bayram":          1 if period_norm == "Bayram" else 0,
        "bayram_period_Bayram_Oncesi":   1 if period_norm == "Bayram_Oncesi" else 0,
        "bayram_period_Bayram_Sonrasi":  1 if period_norm == "Bayram_Sonrasi" else 0,
        "bayram_period_Normal":          1 if period_norm == "Normal" else 0,
        # Weather one-hot
        "weather_condition_Acik":      1 if cond == "Acik" else 0,
        "weather_condition_Ciseleme":  1 if cond == "Ciseleme" else 0,
        "weather_condition_Firtinali": 1 if cond == "Firtinali" else 0,
        "weather_condition_Karli":     1 if cond == "Karli" else 0,
        "weather_condition_Yagmurlu":  1 if cond == "Yagmurlu" else 0,
    }

    # Modelin beklentisine gore kolonlari hizala, eksikleri 0 yap
    cols = feature_cols.get(airline, list(feat.keys()))
    row  = {c: feat.get(c, 0) for c in cols}
    return pd.DataFrame([row])


def predict_airline_price(airline: str, route_code: str, depart_date: date) -> float:
    if airline not in airline_models:
        return 0.0
    X = build_airline_features(airline, route_code, depart_date)
    return max(0.0, float(airline_models[airline].predict(X)[0]))


def predict_all_airlines(route_code: str, depart_date: date) -> dict:
    """Tum havayollari icin tahmin uret. {airline: price}"""
    return {a: round(predict_airline_price(a, route_code, depart_date), 0) for a in AIRLINES}


def get_recommendation(user_price: float, predicted: float, period: str):
    diff_pct = (user_price - predicted) / predicted * 100
    if period == "Bayram":
        if diff_pct <= 5:
            return "AL", "buy", "Bayram dönemi için makul. Daha fazla düşmesi beklenmez."
        elif diff_pct <= 20:
            return "NORMAL", "wait", "Bayram ortalamasında. Acil değilse birkaç gün bekleyebilirsin."
        else:
            return "PAHALI", "expensive", "Bayram ortalamasının üstünde. Kesinlikle bekle."
    else:
        if diff_pct <= -10:
            return "ÇOK UCUZ", "buy", "Ortalamanın çok altında. Kaçırma!"
        elif diff_pct <= 5:
            return "İYİ FİYAT", "buy", "Ortalamanın altında veya civarında. Almak mantıklı."
        elif diff_pct <= 20:
            return "BEKLE", "wait", "Ortalama fiyat. Acil değilse bekle, düşebilir."
        else:
            return "PAHALI", "expensive", "Ortalamanın çok üstünde. Fiyatlar düşene kadar bekle."


def get_price_history(route_code: str, airline: str = None):
    """Gunluk min fiyat gecmisi. Havayolu filtresi opsiyonel."""
    if hist_df.empty:
        return [], []
    r = hist_df[hist_df["route"] == route_code].copy()
    if airline and airline in r.get("airline", pd.Series()).unique():
        r = r[r["airline"] == airline]
    if r.empty:
        return [], []
    daily = r.groupby("depart_date")["price"].min().reset_index()
    daily = daily.sort_values("depart_date")
    return (
        [str(d.date()) for d in daily["depart_date"]],
        [round(float(v), 0) for v in daily["price"]],
    )


def get_alternative_dates(route_code: str, center_date: date, airline: str = None, n: int = 7):
    alts = []
    for delta in range(-n, n + 1):
        d = center_date + timedelta(days=delta)
        if d < date(2026, 5, 1) or d > date(2026, 7, 15):
            continue
        if airline:
            price = predict_airline_price(airline, route_code, d)
        else:
            prices = predict_all_airlines(route_code, d)
            price  = min(prices.values())
        alts.append({
            "date":      d.strftime("%d %b"),
            "date_iso":  str(d),
            "weekday":   WEEKDAY_TR[d.weekday()],
            "price":     round(price, 0),
            "is_selected": (d == center_date),
        })
    if not alts:
        return alts
    min_price = min(a["price"] for a in alts)
    for a in alts:
        if a["price"] == min_price:
            a["tag"] = "En Uygun"
        elif a["price"] <= min_price * 1.05:
            a["tag"] = "İyi Fiyat"
        elif a["is_selected"]:
            a["tag"] = "Mevcut"
        else:
            a["tag"] = "Normal"
    return alts


def get_weekly_forecast(route_code: str, airline: str = None):
    start = date(2026, 5, 1)
    end   = date(2026, 7, 15)
    by_day = {i: {"prices": [], "dates": []} for i in range(7)}
    d = start
    while d <= end:
        if airline:
            price = predict_airline_price(airline, route_code, d)
        else:
            prices = predict_all_airlines(route_code, d)
            price  = min(prices.values())
        by_day[d.weekday()]["prices"].append(price)
        by_day[d.weekday()]["dates"].append(str(d))
        d += timedelta(days=1)
    result = []
    for i in range(7):
        data   = by_day[i]
        prices = data["prices"]
        if prices:
            mid = data["dates"][len(data["dates"]) // 2]
            result.append({"day": WEEKDAY_TR[i], "price": round(float(np.mean(prices)), 0), "date_iso": mid})
        else:
            result.append({"day": WEEKDAY_TR[i], "price": None, "date_iso": None})
    return result


def get_hourly_forecast(route_code: str, target_date: date, airline: str = None):
    if airline:
        base = predict_airline_price(airline, route_code, target_date)
    else:
        prices = predict_all_airlines(route_code, target_date)
        base   = min(prices.values())
    return [{"hour": f"{h:02d}:00", "price": round(base * HOUR_MULTIPLIERS[h], 0)} for h in range(24)]


# ─── Routes ──────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(request, "index.html", {
        "routes":   ROUTES,
        "airlines": AIRLINES,
        "today":    str(date.today()),
        "min_date": "2026-05-01",
        "max_date": "2026-07-15",
    })


@app.post("/predict", response_class=HTMLResponse)
async def predict(
    request:     Request,
    route:       str   = Form(...),
    depart_date: str   = Form(...),
    user_price:  float = Form(...),
    airline:     str   = Form(""),      # bos = tum havayollari
):
    d      = date.fromisoformat(depart_date)
    period = classify_bayram(d)

    # Secilen havayoluna gore tahmin
    selected_airline = airline if airline in AIRLINES else None

    if selected_airline:
        predicted = predict_airline_price(selected_airline, route, d)
    else:
        all_preds = predict_all_airlines(route, d)
        predicted = min(all_preds.values())

    label, status, explanation = get_recommendation(user_price, predicted, period)
    diff     = user_price - predicted
    diff_pct = (diff / predicted) * 100

    # Tum havayolu tahminleri (karsilastirma icin)
    all_preds = predict_all_airlines(route, d)

    hist_labels, hist_values = get_price_history(route, selected_airline)
    alts    = get_alternative_dates(route, d, selected_airline)
    dest    = ROUTE_DEST_CITY[route]
    w_info  = get_weather(dest, d)
    weekly  = get_weekly_forecast(route, selected_airline)

    ctx = {
        "route_label":     ROUTES[route],
        "depart_date":     d.strftime("%d %B %Y"),
        "depart_date_iso": depart_date,
        "route_code":      route,
        "user_price":      user_price,
        "predicted":       round(predicted, 0),
        "diff":            round(diff, 0),
        "diff_pct":        round(diff_pct, 1),
        "label":           label,
        "status":          status,
        "explanation":     explanation,
        "period":          period,
        "selected_airline": selected_airline or "En Uygun",
        "all_predictions": all_preds,   # {airline: price} karsilastirma karti icin
        "hist_labels":     hist_labels,
        "hist_values":     hist_values,
        "alts":            alts,
        "weather":         w_info,
        "city_name":       CITY_NAMES.get(dest, dest),
        "weekly":          weekly,
        "stats":           ROUTE_STATS[route],
        "weekly_labels":   [w["day"] for w in weekly],
        "weekly_values":   [w["price"] for w in weekly],
        "weekly_dates":    [w["date_iso"] for w in weekly],
    }
    return templates.TemplateResponse(request, "partials/result.html", ctx)


@app.get("/hourly", response_class=JSONResponse)
async def hourly(route: str, date_str: str, airline: str = ""):
    d = date.fromisoformat(date_str)
    selected = airline if airline in AIRLINES else None
    data = get_hourly_forecast(route, d, selected)
    return {"hourly": data}


@app.get("/about", response_class=HTMLResponse)
async def about(request: Request):
    return templates.TemplateResponse(request, "about.html", {
        "route_stats": [
            {"label": "İstanbul (IST) → Antalya",  "price": "2.427 TL"},
            {"label": "Antalya → İstanbul (IST)",  "price": "2.215 TL"},
            {"label": "İstanbul (IST) → İzmir",    "price": "1.768 TL"},
            {"label": "İzmir → İstanbul (IST)",    "price": "2.009 TL"},
            {"label": "İstanbul (SAW) → Ankara",   "price": "1.589 TL"},
            {"label": "Ankara → İstanbul (SAW)",   "price": "1.528 TL"},
        ],
        "airline_stats": [
            {"label": "Pegasus", "price": "~1.640 TL"},
            {"label": "AJet",    "price": "~1.800 TL"},
            {"label": "THY",     "price": "~2.900 TL"},
        ],
    })
