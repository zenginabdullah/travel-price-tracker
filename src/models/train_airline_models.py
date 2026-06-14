"""
Havayolu Bazli Model Egitimi
=============================
Her havayolu icin ayri XGBoost + Random Forest modeli egitir.
Toplam 6 model: Pegasus_xgb, Pegasus_rf, AJet_xgb, AJet_rf, THY_xgb, THY_rf

Mantik:
  - Her havayolunun fiyat dinamigi birbirinden cok farkli
  - Pegasus ~2500 TL medyan, THY ~3900 TL medyan
  - Tek modelde bu fark gurultu yaratiyordu

Ciktilar:
  - models/airline/{airline}_{algo}_model.json/joblib
  - reports/model/airline/{airline}_{algo}_results.txt
  - reports/model/airline/{airline}_actual_vs_pred.png
  - reports/model/airline/{airline}_feature_importance.png
  - reports/model/airline/comparison.png
"""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from datetime import date, timedelta
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import optuna
import joblib
import warnings
warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
FLIGHTS_CSV  = PROJECT_ROOT / "data" / "processed" / "flights_enuygun_clean.csv"
WEATHER_CSV  = PROJECT_ROOT / "data" / "processed" / "weather_data.csv"
MODEL_DIR    = PROJECT_ROOT / "models" / "airline"
REPORT_DIR   = PROJECT_ROOT / "reports" / "model" / "airline"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

BAYRAM_START = date(2026, 5, 27)
BAYRAM_END   = date(2026, 5, 30)
DEST_CITY    = {"IST":"IST","AYT":"AYT","ESB":"ESB","ADB":"ADB","SAW":"IST"}
AIRLINES     = ["Pegasus", "AJet", "THY"]
RANDOM_STATE = 42
OPTUNA_TRIALS = 60
PALETTE      = {"Pegasus": "#F97316", "AJet": "#3B82F6", "THY": "#EF4444"}

plt.rcParams["font.family"] = "DejaVu Sans"


# ─── Feature Engineering ─────────────────────────────────────────────────────

def build_features(df: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["depart_date"] = pd.to_datetime(df["depart_date"])
    df["scrape_ts"]   = pd.to_datetime(df["scrape_ts"]).dt.tz_convert("Europe/Istanbul")
    df["scrape_date"] = df["scrape_ts"].dt.normalize().dt.tz_localize(None)
    df["days_to_flight"] = (df["depart_date"] - df["scrape_date"]).dt.days
    df = df[df["days_to_flight"] >= 0].copy()
    df["route"] = df["origin"] + "_" + df["destination"]

    # Saat
    df["departure_hour"] = df["depart_time"].str.split(":").str[0].astype(int, errors="ignore")
    df["is_red_eye"]     = df["departure_hour"].apply(lambda h: 1 if h < 6 or h >= 23 else 0)
    def bucket(h):
        if h < 6:  return "gece"
        if h < 12: return "sabah"
        if h < 17: return "oglen"
        if h < 21: return "aksam"
        return "gec_aksam"
    df["time_of_day"] = df["departure_hour"].apply(bucket)

    # Tarih
    df["depart_weekday"] = df["depart_date"].dt.dayofweek
    df["is_weekend"]     = df["depart_weekday"].isin([5,6]).astype(int)
    df["day_of_month"]   = df["depart_date"].dt.day
    df["week_of_month"]  = (df["day_of_month"] - 1) // 7 + 1
    df["month"]          = df["depart_date"].dt.month

    # Bayram
    d = df["depart_date"].dt.date
    df["is_bayram"]      = d.apply(lambda x: 1 if BAYRAM_START <= x <= BAYRAM_END else 0)
    df["days_to_bayram"] = d.apply(
        lambda x: (BAYRAM_START - x).days if x < BAYRAM_START
        else (x - BAYRAM_END).days if x > BAYRAM_END else 0)
    def period(x):
        if BAYRAM_START <= x <= BAYRAM_END: return "Bayram"
        if (BAYRAM_START - timedelta(7)) <= x < BAYRAM_START: return "Bayram_Oncesi"
        if BAYRAM_END < x <= (BAYRAM_END + timedelta(7)): return "Bayram_Sonrasi"
        return "Normal"
    df["bayram_period"]    = d.apply(period)
    df["bayram_proximity"] = np.exp(-df["days_to_bayram"] / 7)

    # Rota istatistikleri (havayolu bazli)
    for airline in AIRLINES:
        sub = df[df["airline"] == airline]
        stats = sub.groupby("route")["price"].agg(["mean","std"]).rename(
            columns={"mean": f"{airline.lower()}_route_avg",
                     "std":  f"{airline.lower()}_route_std"})
        df = df.merge(stats, on="route", how="left")

    # Genel rota istatistigi
    route_stats = df.groupby("route")["price"].agg(["mean","std"]).rename(
        columns={"mean":"route_global_avg","std":"route_global_std"})
    df = df.merge(route_stats, on="route", how="left")

    # Hava durumu
    if not weather.empty:
        weather["date"] = pd.to_datetime(weather["date"])
        df["dest_city"] = df["destination"].map(DEST_CITY)
        df = df.merge(
            weather[["date","city_code","temp_max","temp_min",
                     "precipitation","wind_speed","weather_condition","is_bad_weather"]],
            left_on=["depart_date","dest_city"], right_on=["date","city_code"], how="left"
        ).drop(columns=["date","city_code","dest_city"], errors="ignore")
        for col in ["temp_max","temp_min","wind_speed"]:
            df[col] = df[col].fillna(df.groupby("destination")[col].transform("median"))
        df["precipitation"]     = df["precipitation"].fillna(0)
        df["weather_condition"] = df["weather_condition"].fillna("Acik")
        df["is_bad_weather"]    = df["is_bad_weather"].fillna(0).astype(int)
    else:
        df["temp_max"]=22.0; df["temp_min"]=12.0
        df["precipitation"]=0.0; df["wind_speed"]=10.0
        df["weather_condition"]="Acik"; df["is_bad_weather"]=0

    return df


def prepare_ml(df: pd.DataFrame):
    """
    Havayolu bazli aggregated min fiyat tahmini.
    Her (rota, gun) icin o havayolunun o gunki min fiyatini hedef al.
    Bu sayede fare-class gurultusu elenir, model gercek taban fiyati ogenir.
    """
    # Rota x gun bazinda aggregasyon
    agg = df.groupby(["route", "depart_date"]).agg(
        min_price       =("price", "min"),
        flight_count    =("price", "count"),
        days_to_flight  =("days_to_flight", "first"),
        depart_weekday  =("depart_weekday", "first"),
        is_weekend      =("is_weekend", "first"),
        day_of_month    =("day_of_month", "first"),
        week_of_month   =("week_of_month", "first"),
        month           =("month", "first"),
        is_bayram       =("is_bayram", "first"),
        days_to_bayram  =("days_to_bayram", "first"),
        bayram_proximity=("bayram_proximity", "first"),
        bayram_period   =("bayram_period", "first"),
        is_red_eye_avail=("is_red_eye", "max"),
        route_global_avg=("route_global_avg", "first"),
        route_global_std=("route_global_std", "first"),
        temp_max        =("temp_max", "first"),
        temp_min        =("temp_min", "first"),
        precipitation   =("precipitation", "first"),
        wind_speed      =("wind_speed", "first"),
        weather_condition=("weather_condition", "first"),
        is_bad_weather  =("is_bad_weather", "first"),
    ).reset_index()

    y = agg["min_price"].copy()
    cat = [c for c in ["route","bayram_period","weather_condition"] if c in agg.columns]
    drop_cols = ["min_price", "depart_date", "route"]
    ml = pd.get_dummies(
        agg.drop(columns=[c for c in drop_cols if c in agg.columns and c not in cat], errors="ignore"),
        columns=cat, drop_first=False, dtype=int
    )
    ml = ml.drop(columns=["min_price"], errors="ignore")
    return ml, y


# ─── Optuna ──────────────────────────────────────────────────────────────────

def xgb_objective(trial, X_tr, y_tr):
    p = {
        "n_estimators":     trial.suggest_int("n_estimators", 100, 600),
        "max_depth":        trial.suggest_int("max_depth", 3, 7),
        "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "subsample":        trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha":        trial.suggest_float("reg_alpha", 0.0, 5.0),
        "reg_lambda":       trial.suggest_float("reg_lambda", 1.0, 10.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 3, 20),
        "gamma":            trial.suggest_float("gamma", 0.0, 3.0),
    }
    m  = XGBRegressor(**p, random_state=RANDOM_STATE, n_jobs=-1)
    cv = KFold(5, shuffle=True, random_state=RANDOM_STATE)
    return -cross_val_score(m, X_tr, y_tr, cv=cv, scoring="neg_mean_absolute_error").mean()


def rf_objective(trial, X_tr, y_tr):
    p = {
        "n_estimators":     trial.suggest_int("n_estimators", 100, 500),
        "max_depth":        trial.suggest_int("max_depth", 4, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
        "max_features":     trial.suggest_float("max_features", 0.3, 1.0),
    }
    m  = RandomForestRegressor(**p, random_state=RANDOM_STATE, n_jobs=-1)
    cv = KFold(5, shuffle=True, random_state=RANDOM_STATE)
    return -cross_val_score(m, X_tr, y_tr, cv=cv, scoring="neg_mean_absolute_error").mean()


# ─── Grafikler ───────────────────────────────────────────────────────────────

ALGO_STYLE = {
    "XGBoost":       {"marker": "o", "linestyle": "-",  "hatch": ""},
    "Random Forest": {"marker": "s", "linestyle": "--", "hatch": "//"},
}

def plot_all_results(all_results: list):
    """
    3 kapsamli grafik dosyasi uretir:
      01_actual_vs_pred.png     - 3 havayolu × 2 algo, tek grafik (farkli renkler/marker)
      02_residuals.png          - 3 havayolu × 2 algo residual dagilimi
      03_metrics_comparison.png - CV R2 / MAE / MAPE karsilastirma + feature importance
    """
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D

    # ── Ortak legend elementi ─────────────────────────────────────────────────
    airline_patches = [Patch(fc=PALETTE[a], label=a) for a in AIRLINES]
    algo_lines = [
        Line2D([0],[0], color="k", ls="-",  marker="o", ms=6, label="XGBoost"),
        Line2D([0],[0], color="k", ls="--", marker="s", ms=6, label="Random Forest"),
    ]

    # ── 1) Actual vs Predicted — tum modeller tek grafik ─────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, airline in zip(axes, AIRLINES):
        color = PALETTE[airline]
        airline_res = [r for r in all_results if r["airline"] == airline]
        all_vals = []
        for r in airline_res:
            all_vals += list(r["y_te"]) + list(r["y_pred"])
        mn, mx = min(all_vals) - 100, max(all_vals) + 100
        ax.plot([mn, mx], [mn, mx], "k--", linewidth=1.2, alpha=0.4, zorder=0)
        for r in airline_res:
            style = ALGO_STYLE[r["algo"]]
            ax.scatter(r["y_te"], r["y_pred"],
                       alpha=0.45, s=20, color=color,
                       marker=style["marker"],
                       edgecolors="white", linewidth=0.2,
                       label=r["algo"])
            ax.text(0.05, 0.97 - airline_res.index(r) * 0.12,
                    f"{r['algo'][:3]}: R2={r['test_r2']:.3f}  MAE={r['test_mae']:.0f}TL",
                    transform=ax.transAxes, fontsize=8, va="top",
                    color=color,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7, ec=color))
        ax.set_xlabel("Gercek Min Fiyat (TL)"); ax.set_ylabel("Tahmin (TL)")
        ax.set_title(airline, fontsize=12, fontweight="bold", color=color)
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)
    fig.suptitle("Actual vs Predicted — Havayolu Bazli", fontsize=14, fontweight="bold")
    fig.legend(handles=airline_patches + algo_lines, loc="lower center",
               ncol=5, bbox_to_anchor=(0.5, -0.06), fontsize=9)
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "01_actual_vs_pred.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ── 2) Residuals — 3×2 grid ───────────────────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(18, 9))
    for col, airline in enumerate(AIRLINES):
        color = PALETTE[airline]
        for row, algo in enumerate(["XGBoost", "Random Forest"]):
            r = next((x for x in all_results if x["airline"] == airline and x["algo"] == algo), None)
            ax = axes[row][col]
            if r is None:
                ax.axis("off"); continue
            res = r["y_te"].values - r["y_pred"]
            ax.scatter(r["y_pred"], res, alpha=0.4, s=18, color=color,
                       marker=ALGO_STYLE[algo]["marker"], edgecolors="white", linewidth=0.2)
            ax.axhline(0, color="black", linestyle="--", linewidth=1.3)
            ax.axhline( res.std(), color=color, linestyle=":", linewidth=1, alpha=0.6)
            ax.axhline(-res.std(), color=color, linestyle=":", linewidth=1, alpha=0.6)
            ax.set_xlabel("Tahmin (TL)"); ax.set_ylabel("Hata (TL)")
            ax.set_title(f"{airline} — {algo}", fontweight="bold", color=color)
            ax.text(0.02, 0.97, f"std={res.std():.0f} TL\nmean={res.mean():.0f} TL",
                    transform=ax.transAxes, fontsize=8, va="top",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7))
            ax.grid(True, alpha=0.25)
    fig.suptitle("Residual Dagilimi — Havayolu × Algoritma", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "02_residuals.png", dpi=150)
    plt.close()

    # ── 3) Metrik karsilastirma + Feature Importance ──────────────────────────
    fig = plt.figure(figsize=(20, 10))
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    labels  = [f"{r['airline']}\n{r['algo'][:3]}" for r in all_results]
    colors  = [PALETTE.get(r["airline"], "#6366F1") for r in all_results]
    hatches = [ALGO_STYLE[r["algo"]]["hatch"] for r in all_results]
    x       = range(len(all_results))

    def bar_chart(ax, values, ylabel, title, fmt="{:.3f}", ref=None):
        bars = ax.bar(x, values, color=colors, edgecolor="white", width=0.6)
        for bar, h, v in zip(bars, hatches, values):
            bar.set_hatch(h)
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(values)*0.01,
                    fmt.format(v), ha="center", fontsize=8)
        if ref: ax.axhline(ref, color="gray", linestyle="--", alpha=0.5, linewidth=1)
        ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylabel(ylabel); ax.set_title(title, fontweight="bold")
        ax.grid(True, alpha=0.2, axis="y")

    ax1 = fig.add_subplot(gs[0, 0])
    bar_chart(ax1, [r["cv_r2"]  for r in all_results], "CV R2",    "CV R2 (Yuksek=Iyi)", "{:.3f}", ref=0.7)
    ax1.set_ylim(0, 1)

    ax2 = fig.add_subplot(gs[0, 1])
    bar_chart(ax2, [r["cv_mae"] for r in all_results], "MAE (TL)", "CV MAE (Dusuk=Iyi)", "{:.0f}")

    ax3 = fig.add_subplot(gs[0, 2])
    bar_chart(ax3, [r["test_mape"] for r in all_results], "MAPE (%)", "Test MAPE (Dusuk=Iyi)", "{:.1f}%")

    # Feature importance — her havayolu icin XGBoost + RF yan yana
    gs2 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[1, :], wspace=0.05)
    gs2_cols = [gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs2[0], wspace=0.4),
                gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs2[1], wspace=0.4)]
    for col, airline in enumerate(AIRLINES):
        color = PALETTE[airline]
        for row, algo in enumerate(["XGBoost", "Random Forest"]):
            ax = fig.add_subplot(gs2_cols[row][col])
            r = next((x for x in all_results if x["airline"] == airline and x["algo"] == algo), None)
            if r and r.get("feat_imp") is not None:
                top = r["feat_imp"].head(10)
                ax.barh(range(len(top)), top["importance"],
                        color=color, alpha=0.7 if algo == "Random Forest" else 0.9,
                        edgecolor="white",
                        hatch=ALGO_STYLE[algo]["hatch"])
                ax.set_yticks(range(len(top))); ax.set_yticklabels(top["feature"], fontsize=7)
                ax.invert_yaxis()
                ax.set_xlabel("Importance", fontsize=7)
                ax.set_title(f"{airline}\n{algo[:3]}", fontweight="bold", color=color, fontsize=8)
                ax.grid(True, alpha=0.2, axis="x")
                ax.tick_params(axis="x", labelsize=7)
            else:
                ax.axis("off")

    # Legend
    fig.legend(handles=airline_patches + algo_lines, loc="lower center",
               ncol=5, bbox_to_anchor=(0.5, -0.03), fontsize=9)
    fig.suptitle("Havayolu Bazli Model Degerlendirmesi", fontsize=15, fontweight="bold")
    plt.savefig(REPORT_DIR / "03_metrics_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()


# ─── Ana Fonksiyon ────────────────────────────────────────────────────────────

def remove_outliers_iqr(df: pd.DataFrame, group_col: str, price_col: str = "price") -> pd.DataFrame:
    """Rota bazli IQR yontemiyle outlier temizle (Q3 + 1.5*IQR ustunu cikar)."""
    masks = []
    for _, g in df.groupby(group_col):
        Q1, Q3 = g[price_col].quantile(0.25), g[price_col].quantile(0.75)
        upper = Q3 + 1.5 * (Q3 - Q1)
        masks.append(g.index[g[price_col] <= upper])
    keep = [idx for m in masks for idx in m]
    return df.loc[keep].reset_index(drop=True)


def train_airline(airline: str, df_feat: pd.DataFrame) -> list:
    results = []
    sub = df_feat[df_feat["airline"] == airline].copy()

    # Rota bazli IQR outlier temizligi (tum havayollari)
    before = len(sub)
    sub = remove_outliers_iqr(sub, group_col="route")
    print(f"  [Outlier] {before:,} -> {len(sub):,} kayit ({before-len(sub)} cikarildi)")

    print(f"\n{'='*55}")
    print(f"  {airline}  ({len(sub):,} ucus)")
    print(f"{'='*55}")

    X, y = prepare_ml(sub)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
    cv = KFold(5, shuffle=True, random_state=RANDOM_STATE)

    # ── XGBoost ──────────────────────────────────────────────────────────────
    print(f"  XGBoost Optuna ({OPTUNA_TRIALS} trial)...")
    study_xgb = optuna.create_study(direction="minimize")
    study_xgb.optimize(lambda t: xgb_objective(t, X_tr, y_tr), n_trials=OPTUNA_TRIALS)
    best_xgb = study_xgb.best_params

    xgb = XGBRegressor(**best_xgb, random_state=RANDOM_STATE, n_jobs=-1)
    xgb.fit(X_tr, y_tr)
    y_pred_xgb = xgb.predict(X_te)

    cv_mae_xgb = -cross_val_score(xgb, X, y, cv=cv, scoring="neg_mean_absolute_error")
    cv_r2_xgb  =  cross_val_score(xgb, X, y, cv=cv, scoring="r2")
    mape_xgb   = np.mean(np.abs((y_te - y_pred_xgb) / y_te)) * 100

    feat_imp = pd.DataFrame({"feature": X.columns, "importance": xgb.feature_importances_}
                             ).sort_values("importance", ascending=False)
    xgb.save_model(str(MODEL_DIR / f"{airline.lower()}_xgb_model.json"))

    print(f"  XGB  CV R2={cv_r2_xgb.mean():.4f}  MAE={cv_mae_xgb.mean():.0f} TL  MAPE={mape_xgb:.1f}%")
    results.append({"airline": airline, "algo": "XGBoost",
                    "cv_r2": cv_r2_xgb.mean(), "cv_mae": cv_mae_xgb.mean(),
                    "test_r2": r2_score(y_te, y_pred_xgb),
                    "test_mae": mean_absolute_error(y_te, y_pred_xgb),
                    "test_mape": mape_xgb, "params": best_xgb,
                    "feature_cols": list(X.columns),
                    "y_te": y_te, "y_pred": y_pred_xgb, "feat_imp": feat_imp})

    # ── Random Forest ─────────────────────────────────────────────────────────
    print(f"  Random Forest Optuna ({OPTUNA_TRIALS} trial)...")
    study_rf = optuna.create_study(direction="minimize")
    study_rf.optimize(lambda t: rf_objective(t, X_tr, y_tr), n_trials=OPTUNA_TRIALS)
    best_rf = study_rf.best_params

    rf = RandomForestRegressor(**best_rf, random_state=RANDOM_STATE, n_jobs=-1)
    rf.fit(X_tr, y_tr)
    y_pred_rf = rf.predict(X_te)

    cv_mae_rf = -cross_val_score(rf, X, y, cv=cv, scoring="neg_mean_absolute_error")
    cv_r2_rf  =  cross_val_score(rf, X, y, cv=cv, scoring="r2")
    mape_rf   = np.mean(np.abs((y_te - y_pred_rf) / y_te)) * 100

    rf_feat_imp = pd.DataFrame({"feature": X.columns, "importance": rf.feature_importances_}
                                ).sort_values("importance", ascending=False)
    joblib.dump(rf, MODEL_DIR / f"{airline.lower()}_rf_model.joblib")

    print(f"  RF   CV R2={cv_r2_rf.mean():.4f}  MAE={cv_mae_rf.mean():.0f} TL  MAPE={mape_rf:.1f}%")
    results.append({"airline": airline, "algo": "Random Forest",
                    "cv_r2": cv_r2_rf.mean(), "cv_mae": cv_mae_rf.mean(),
                    "test_r2": r2_score(y_te, y_pred_rf),
                    "test_mae": mean_absolute_error(y_te, y_pred_rf),
                    "test_mape": mape_rf, "params": best_rf,
                    "feature_cols": list(X.columns),
                    "y_te": y_te, "y_pred": y_pred_rf, "feat_imp": rf_feat_imp})

    # Rapor kaydet
    lines = [
        f"{'='*55}", f"Havayolu: {airline}", f"{'='*55}",
        f"Veri: {len(sub):,} ucus kaydı",
        f"Tarih: {sub['depart_date'].min().date()} - {sub['depart_date'].max().date()}",
        "", "XGBoost:",
        f"  CV R2:  {cv_r2_xgb.mean():.4f} +/- {cv_r2_xgb.std():.4f}",
        f"  CV MAE: {cv_mae_xgb.mean():.0f} +/- {cv_mae_xgb.std():.0f} TL",
        f"  Test R2: {r2_score(y_te, y_pred_xgb):.4f}  MAE: {mean_absolute_error(y_te, y_pred_xgb):.0f} TL  MAPE: {mape_xgb:.1f}%",
        f"  Params: {best_xgb}",
        "", "Random Forest:",
        f"  CV R2:  {cv_r2_rf.mean():.4f} +/- {cv_r2_rf.std():.4f}",
        f"  CV MAE: {cv_mae_rf.mean():.0f} +/- {cv_mae_rf.std():.0f} TL",
        f"  Test R2: {r2_score(y_te, y_pred_rf):.4f}  MAE: {mean_absolute_error(y_te, y_pred_rf):.0f} TL  MAPE: {mape_rf:.1f}%",
        f"  Params: {best_rf}",
        "", "Top 10 Feature (XGBoost):",
    ] + [f"  {row['feature']:<35} {row['importance']:.4f}" for _, row in feat_imp.head(10).iterrows()]
    (REPORT_DIR / f"{airline.lower()}_results.txt").write_text("\n".join(lines), encoding="utf-8")

    return results


def main():
    print("Havayolu bazli model egitimi basliyor...\n")

    df = pd.read_csv(FLIGHTS_CSV)
    weather = pd.read_csv(WEATHER_CSV) if WEATHER_CSV.exists() else pd.DataFrame()
    print(f"Ham kayit: {len(df):,}")

    print("\nFeature engineering...")
    df_feat = build_features(df, weather)
    print(f"Feature kayit: {len(df_feat):,}, {df_feat.shape[1]} sutun")

    all_results = []
    for airline in AIRLINES:
        res = train_airline(airline, df_feat)
        all_results.extend(res)

    print("\n\nGrafikler uretiliyor...")
    plot_all_results(all_results)

    # Genel ozet
    print(f"\n{'='*55}")
    print("GENEL OZET")
    print(f"{'='*55}")
    print(f"{'Havayolu':<12} {'Algo':<14} {'CV R2':>8} {'CV MAE':>10} {'MAPE':>8}")
    print(f"{'─'*12} {'─'*14} {'─'*8} {'─'*10} {'─'*8}")
    for r in all_results:
        print(f"  {r['airline']:<10} {r['algo']:<14} {r['cv_r2']:>8.4f} {r['cv_mae']:>8.0f} TL {r['test_mape']:>6.1f}%")

    # JSON olarak feature kolonlari kaydet (inference icin)
    import json
    feature_map = {r["airline"]: r["feature_cols"]
                   for r in all_results if r["algo"] == "XGBoost"}
    (MODEL_DIR / "feature_columns.json").write_text(
        json.dumps(feature_map, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nModel dosyalari: {MODEL_DIR}")
    print(f"Rapor dosyalari: {REPORT_DIR}")
    print("\nTamamlandi!")


if __name__ == "__main__":
    main()
