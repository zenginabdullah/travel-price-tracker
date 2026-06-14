"""
EDA (Exploratory Data Analysis) - Kesifsel Veri Analizi
========================================================
Grafik listesi:
  01_price_distribution.png    - Fiyat dagilimi histogram + boxplot
  02_price_by_route.png        - Rota bazli fiyat karsilastirmasi
  03_price_by_airline.png      - Havayolu bazli fiyat karsilastirmasi
  04_bayram_effect.png         - Bayram donemi fiyat etkisi
  05_days_to_flight.png        - Uçusa kalan gun vs fiyat
  06_time_weekday.png          - Saat dilimi ve hafta gunu etkisi
  07_correlation_heatmap.png   - Korelasyon matrisi
  08_price_trend_by_route.png  - Rota bazli fiyat trendi zaman icinde
  09_airline_route_heatmap.png - Havayolu x rota fiyat heatmap
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from datetime import date

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
INPUT_CSV    = PROJECT_ROOT / "data" / "processed" / "flights_features.csv"
OUTPUT_DIR   = PROJECT_ROOT / "reports" / "eda"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams["font.family"] = "DejaVu Sans"
sns.set_style("whitegrid")
PALETTE = {"Pegasus": "#F97316", "AJet": "#3B82F6", "THY": "#EF4444"}


def load_data() -> pd.DataFrame:
    df = pd.read_csv(INPUT_CSV)
    df["depart_date"] = pd.to_datetime(df["depart_date"])
    # bayram_period degerlerini standartlastir
    period_map = {
        "Pre-Bayram": "Bayram Oncesi", "Post-Bayram": "Bayram Sonrasi",
        "Bayram Oncesi": "Bayram Oncesi", "Bayram Sonrasi": "Bayram Sonrasi",
        "Bayram": "Bayram", "Normal": "Normal",
    }
    if "bayram_period" in df.columns:
        df["bayram_period"] = df["bayram_period"].map(period_map).fillna("Normal")
    return df


# ── 01: Fiyat Dagilimi ────────────────────────────────────────────────────────
def plot_price_distribution(df: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].hist(df["price"], bins=60, color="#3B82F6", edgecolor="white", alpha=0.85)
    axes[0].axvline(df["price"].mean(),   color="#EF4444", linestyle="--", linewidth=1.8,
                    label=f'Ortalama: {df["price"].mean():,.0f} TL')
    axes[0].axvline(df["price"].median(), color="#F97316", linestyle="--", linewidth=1.8,
                    label=f'Medyan: {df["price"].median():,.0f} TL')
    axes[0].set_xlabel("Fiyat (TL)"); axes[0].set_ylabel("Frekans")
    axes[0].set_title("Fiyat Dagilimi")
    axes[0].legend()

    axes[1].boxplot(df["price"], vert=True, patch_artist=True,
                    boxprops=dict(facecolor="#93C5FD", color="#1D4ED8"),
                    medianprops=dict(color="#EF4444", linewidth=2))
    axes[1].set_ylabel("Fiyat (TL)")
    axes[1].set_title(f"Fiyat Boxplot  (N={len(df):,})")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "01_price_distribution.png", dpi=150)
    plt.close()


# ── 02: Rota Bazli ────────────────────────────────────────────────────────────
def plot_price_by_route(df: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    route_order = df.groupby("route")["price"].median().sort_values().index
    sns.boxplot(data=df, x="route", y="price", order=route_order,
                ax=axes[0], palette="Set2", showfliers=False)
    axes[0].set_xlabel("Rota"); axes[0].set_ylabel("Fiyat (TL)")
    axes[0].set_title("Rota Bazli Fiyat Dagilimi (outlier gizli)")
    axes[0].tick_params(axis="x", rotation=30)

    route_stats = df.groupby("route")["price"].agg(["median","mean","min"]).loc[route_order]
    x = range(len(route_stats))
    axes[1].bar(x, route_stats["median"], color="#93C5FD", label="Medyan")
    axes[1].bar(x, route_stats["min"],    color="#6EE7B7", label="Min", alpha=0.8)
    axes[1].plot(x, route_stats["mean"],  "o--", color="#EF4444", label="Ortalama")
    axes[1].set_xticks(x); axes[1].set_xticklabels(route_order, rotation=30)
    axes[1].set_ylabel("Fiyat (TL)")
    axes[1].set_title("Rota Bazli Min / Medyan / Ortalama")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "02_price_by_route.png", dpi=150)
    plt.close()


# ── 03: Havayolu Bazli ────────────────────────────────────────────────────────
def plot_price_by_airline(df: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    airline_order = df.groupby("airline")["price"].median().sort_values().index
    colors = [PALETTE.get(a, "#94A3B8") for a in airline_order]

    sns.boxplot(data=df, x="airline", y="price", order=airline_order,
                ax=axes[0], palette=colors, showfliers=False)
    axes[0].set_xlabel("Havayolu"); axes[0].set_ylabel("Fiyat (TL)")
    axes[0].set_title("Havayolu Bazli Fiyat Dagilimi")

    # Ucus sayisi
    counts = df["airline"].value_counts().loc[airline_order]
    axes[1].bar(range(len(counts)), counts.values, color=colors, edgecolor="white")
    axes[1].set_xticks(range(len(counts))); axes[1].set_xticklabels(counts.index)
    axes[1].set_ylabel("Ucus Sayisi"); axes[1].set_title("Havayolu Ucus Sayisi")
    for i, v in enumerate(counts.values):
        axes[1].text(i, v + 20, str(v), ha="center", fontsize=10)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "03_price_by_airline.png", dpi=150)
    plt.close()


# ── 04: Bayram Etkisi ─────────────────────────────────────────────────────────
def plot_bayram_effect(df: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    period_order  = ["Normal", "Bayram Oncesi", "Bayram", "Bayram Sonrasi"]
    period_colors = ["#4ECDC4", "#FFD93D", "#FF6B6B", "#95E1D3"]
    plot_df = df[df["bayram_period"].isin(period_order)]
    sns.boxplot(data=plot_df, x="bayram_period", y="price", order=period_order,
                ax=axes[0], palette=period_colors, showfliers=False)
    axes[0].set_xlabel("Donem"); axes[0].set_ylabel("Fiyat (TL)")
    axes[0].set_title("Bayram Donemine Gore Fiyatlar")
    axes[0].tick_params(axis="x", rotation=15)

    daily_avg = df.groupby("depart_date")["price"].mean().reset_index()
    axes[1].plot(daily_avg["depart_date"], daily_avg["price"],
                 linewidth=1.5, color="#3B82F6", label="Gunluk Ort.")
    axes[1].axvspan(pd.Timestamp("2026-05-27"), pd.Timestamp("2026-05-30"),
                    alpha=0.25, color="#EF4444", label="Bayram")
    axes[1].axvspan(pd.Timestamp("2026-05-20"), pd.Timestamp("2026-05-27"),
                    alpha=0.10, color="#F97316", label="Bayram Oncesi")
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    axes[1].xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
    axes[1].set_xlabel("Kalkis Tarihi"); axes[1].set_ylabel("Ortalama Fiyat (TL)")
    axes[1].set_title("Gunluk Ortalama Fiyat Trendi")
    axes[1].legend(); axes[1].tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "04_bayram_effect.png", dpi=150)
    plt.close()


# ── 05: Booking Zamanlama ─────────────────────────────────────────────────────
def plot_days_to_flight(df: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    trend = df.groupby("days_to_flight")["price"].agg(["mean","median","std"]).reset_index()
    trend = trend[trend["days_to_flight"] <= 90]

    axes[0].plot(trend["days_to_flight"], trend["median"],
                 linewidth=2, color="#3B82F6", label="Medyan")
    axes[0].fill_between(trend["days_to_flight"],
                         trend["median"] - trend["std"] * 0.5,
                         trend["median"] + trend["std"] * 0.5,
                         alpha=0.15, color="#3B82F6")
    axes[0].set_xlabel("Ucusa Kalan Gun"); axes[0].set_ylabel("Fiyat (TL)")
    axes[0].set_title("Erken Rezervasyon vs Fiyat")
    axes[0].invert_xaxis(); axes[0].legend(); axes[0].grid(True, alpha=0.3)

    # Havayolu bazli ayni grafik
    for airline, color in PALETTE.items():
        sub = df[df["airline"] == airline]
        t2 = sub.groupby("days_to_flight")["price"].median().reset_index()
        t2 = t2[t2["days_to_flight"] <= 90]
        axes[1].plot(t2["days_to_flight"], t2["price"],
                     linewidth=1.8, color=color, label=airline)
    axes[1].set_xlabel("Ucusa Kalan Gun"); axes[1].set_ylabel("Medyan Fiyat (TL)")
    axes[1].set_title("Havayolu Bazli Erken Rezervasyon Etkisi")
    axes[1].invert_xaxis(); axes[1].legend(); axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "05_days_to_flight.png", dpi=150)
    plt.close()


# ── 06: Saat & Hafta Gunu ─────────────────────────────────────────────────────
def plot_time_and_weekday(df: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    if "depart_hour" in df.columns:
        hour_med = df.groupby("depart_hour")["price"].median().reset_index()
        axes[0].bar(hour_med["depart_hour"], hour_med["price"],
                    color="#3B82F6", edgecolor="white", alpha=0.85)
        axes[0].set_xlabel("Kalkis Saati (0-23)"); axes[0].set_ylabel("Medyan Fiyat (TL)")
        axes[0].set_title("Saate Gore Medyan Fiyat")
        axes[0].set_xticks(range(0, 24, 2))
    else:
        time_order = ["Sabah", "Oglen", "Aksam", "Gece"]
        time_df = df[df.get("time_of_day", pd.Series()).isin(time_order)] if "time_of_day" in df.columns else df
        if "time_of_day" in df.columns:
            sns.boxplot(data=time_df, x="time_of_day", y="price",
                        order=[t for t in time_order if t in time_df["time_of_day"].unique()],
                        ax=axes[0], palette=["#FFD700","#FF6347","#4169E1","#191970"],
                        showfliers=False)
        axes[0].set_title("Saat Dilimine Gore Fiyat")

    weekday_names = ["Pzt","Sal","Car","Per","Cum","Cmt","Paz"]
    weekday_med = df.groupby("depart_weekday")["price"].median().reset_index()
    colors_wd = ["#EF4444" if d in [5,6] else "#3B82F6" for d in weekday_med["depart_weekday"]]
    axes[1].bar(weekday_med["depart_weekday"], weekday_med["price"],
                color=colors_wd, edgecolor="white")
    axes[1].set_xticks(range(7)); axes[1].set_xticklabels(weekday_names)
    axes[1].set_xlabel("Hafta Gunu"); axes[1].set_ylabel("Medyan Fiyat (TL)")
    axes[1].set_title("Hafta Gunune Gore Fiyat (Kirmizi=Hafta Sonu)")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "06_time_weekday.png", dpi=150)
    plt.close()


# ── 07: Korelasyon Heatmap ────────────────────────────────────────────────────
def plot_correlation_heatmap(df: pd.DataFrame):
    numeric_cols = ["price","days_to_flight","depart_weekday","is_weekend",
                    "day_of_month","depart_hour","is_bayram","days_to_bayram",
                    "temp_max","precipitation","is_bad_weather"]
    corr_cols = [c for c in numeric_cols if c in df.columns]
    corr = df[corr_cols].corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
                center=0, ax=ax, square=True, linewidths=0.5,
                annot_kws={"size": 9})
    ax.set_title("Feature Korelasyon Matrisi")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "07_correlation_heatmap.png", dpi=150)
    plt.close()
    return corr


# ── 08: Rota Bazli Fiyat Trendi ───────────────────────────────────────────────
def plot_price_trend_by_route(df: pd.DataFrame):
    routes = df["route"].unique()
    ncols = 2
    nrows = -(-len(routes) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, nrows * 4), squeeze=False)

    for idx, route in enumerate(sorted(routes)):
        ax = axes[idx // ncols][idx % ncols]
        sub = df[df["route"] == route]
        daily = sub.groupby("depart_date")["price"].agg(["min","median"]).reset_index()
        ax.plot(daily["depart_date"], daily["median"],
                linewidth=1.5, color="#3B82F6", label="Medyan")
        ax.plot(daily["depart_date"], daily["min"],
                linewidth=1.5, color="#10B981", linestyle="--", label="Min")
        ax.axvspan(pd.Timestamp("2026-05-27"), pd.Timestamp("2026-05-30"),
                   alpha=0.2, color="#EF4444")
        ax.set_title(route); ax.set_ylabel("Fiyat (TL)")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
        ax.tick_params(axis="x", rotation=45)
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    for idx in range(len(routes), nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    fig.suptitle("Rota Bazli Fiyat Trendi (Kirmizi=Bayram)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "08_price_trend_by_route.png", dpi=150)
    plt.close()


# ── 09: Havayolu x Rota Heatmap ───────────────────────────────────────────────
def plot_airline_route_heatmap(df: pd.DataFrame):
    pivot = df.groupby(["airline","route"])["price"].median().unstack(fill_value=0)

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    sns.heatmap(pivot, annot=True, fmt=",.0f", cmap="YlOrRd",
                ax=axes[0], linewidths=0.5, annot_kws={"size": 9})
    axes[0].set_title("Havayolu x Rota Medyan Fiyat (TL)")
    axes[0].set_xlabel("Rota"); axes[0].set_ylabel("Havayolu")
    axes[0].tick_params(axis="x", rotation=30)

    pivot_min = df.groupby(["airline","route"])["price"].min().unstack(fill_value=0)
    sns.heatmap(pivot_min, annot=True, fmt=",.0f", cmap="YlGn",
                ax=axes[1], linewidths=0.5, annot_kws={"size": 9})
    axes[1].set_title("Havayolu x Rota Min Fiyat (TL)")
    axes[1].set_xlabel("Rota"); axes[1].set_ylabel("Havayolu")
    axes[1].tick_params(axis="x", rotation=30)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "09_airline_route_heatmap.png", dpi=150)
    plt.close()


# ── Ozet ─────────────────────────────────────────────────────────────────────
def generate_summary(df: pd.DataFrame, corr: pd.DataFrame):
    lines = [
        "=" * 60,
        "EDA OZET RAPORU",
        "=" * 60,
        f"\nDataset : {len(df):,} satir, {len(df.columns)} sutun",
        f"Tarih   : {df['depart_date'].min().date()} - {df['depart_date'].max().date()}",
        f"Rota    : {df['route'].nunique()}",
        f"Havayolu: {df['airline'].nunique()}",
        "",
        "─" * 60,
        "FIYAT ISTATISTIKLERI",
        "─" * 60,
        f"  Min      : {df['price'].min():>8,.0f} TL",
        f"  Q1       : {df['price'].quantile(.25):>8,.0f} TL",
        f"  Medyan   : {df['price'].median():>8,.0f} TL",
        f"  Ortalama : {df['price'].mean():>8,.0f} TL",
        f"  Q3       : {df['price'].quantile(.75):>8,.0f} TL",
        f"  Max      : {df['price'].max():>8,.0f} TL",
        f"  Std      : {df['price'].std():>8,.0f} TL",
        "",
        "─" * 60,
        "HAVAYOLU BAZLI",
        "─" * 60,
    ]
    for airline in df["airline"].unique():
        sub = df[df["airline"] == airline]
        lines.append(f"  {airline:<10} Medyan:{sub['price'].median():>7,.0f} TL  "
                     f"Min:{sub['price'].min():>7,.0f} TL  N={len(sub):,}")

    lines += ["", "─"*60, "BAYRAM ETKISI", "─"*60]
    for period in ["Normal","Bayram Oncesi","Bayram","Bayram Sonrasi"]:
        sub = df[df["bayram_period"] == period] if "bayram_period" in df.columns else pd.DataFrame()
        if not sub.empty:
            lines.append(f"  {period:<16} Medyan:{sub['price'].median():>7,.0f} TL  N={len(sub):,}")

    if "bayram_period" in df.columns:
        n_med = df[df["bayram_period"] == "Normal"]["price"].median()
        b_med = df[df["bayram_period"] == "Bayram"]["price"].median()
        if n_med > 0:
            lines.append(f"\n  Bayram fiyat artisi: %{(b_med-n_med)/n_med*100:.1f} (medyan bazli)")

    lines += ["", "─"*60, "PRICE KORELASYONLARI (top 8)", "─"*60]
    if "price" in corr.columns:
        top_corr = corr["price"].drop("price").abs().sort_values(ascending=False).head(8)
        for feat in top_corr.index:
            lines.append(f"  {feat:<25} {corr['price'][feat]:+.3f}")

    summary = "\n".join(lines)
    (OUTPUT_DIR / "00_eda_summary.txt").write_text(summary, encoding="utf-8")
    print(summary)


def main():
    print("EDA basliyor...\n")
    df = load_data()
    print(f"  {len(df):,} satir yuklendi")

    print("[1/9] Fiyat dagilimi...")
    plot_price_distribution(df)

    print("[2/9] Rota bazli fiyatlar...")
    plot_price_by_route(df)

    print("[3/9] Havayolu bazli fiyatlar...")
    plot_price_by_airline(df)

    print("[4/9] Bayram etkisi...")
    plot_bayram_effect(df)

    print("[5/9] Booking zamanlama...")
    plot_days_to_flight(df)

    print("[6/9] Saat & hafta gunu...")
    plot_time_and_weekday(df)

    print("[7/9] Korelasyon matrisi...")
    corr = plot_correlation_heatmap(df)

    print("[8/9] Rota bazli trend...")
    plot_price_trend_by_route(df)

    print("[9/9] Havayolu x rota heatmap...")
    plot_airline_route_heatmap(df)

    print("\n" + "="*60)
    generate_summary(df, corr)
    print(f"\nEDA tamamlandi! --> {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
