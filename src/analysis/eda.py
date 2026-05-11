"""
EDA (Exploratory Data Analysis) - Keşifsel Veri Analizi
========================================================
Veriyi görselleştirip temel istatistiksel ilişkileri ortaya koyar.
Çıktılar: reports/eda/ altına PNG grafikler + özet txt.
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ─── Yollar ───────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
INPUT_CSV = PROJECT_ROOT / "data" / "processed" / "flights_features.csv"
OUTPUT_DIR = PROJECT_ROOT / "reports" / "eda"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Türkçe karakter desteği için
plt.rcParams["font.family"] = "DejaVu Sans"
sns.set_style("whitegrid")


def load_data() -> pd.DataFrame:
    df = pd.read_csv(INPUT_CSV)
    df["depart_date"] = pd.to_datetime(df["depart_date"])
    return df


def plot_price_distribution(df: pd.DataFrame):
    """1. Fiyat dağılımı histogramı + boxplot."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Histogram
    axes[0].hist(df["price"], bins=50, color="#2E86C1", edgecolor="white", alpha=0.8)
    axes[0].axvline(df["price"].mean(), color="red", linestyle="--", label=f'Ortalama: {df["price"].mean():.0f}₺')
    axes[0].axvline(df["price"].median(), color="orange", linestyle="--", label=f'Medyan: {df["price"].median():.0f}₺')
    axes[0].set_xlabel("Fiyat (TL)")
    axes[0].set_ylabel("Frekans")
    axes[0].set_title("Fiyat Dağılımı")
    axes[0].legend()

    # Boxplot
    axes[1].boxplot(df["price"], vert=True)
    axes[1].set_ylabel("Fiyat (TL)")
    axes[1].set_title("Fiyat Boxplot (Outlier Görünümü)")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "01_price_distribution.png", dpi=150)
    plt.close()


def plot_price_by_route(df: pd.DataFrame):
    """2. Rota bazlı fiyat karşılaştırması."""
    fig, ax = plt.subplots(figsize=(12, 6))

    route_order = df.groupby("route")["price"].median().sort_values().index
    sns.boxplot(data=df, x="route", y="price", order=route_order, ax=ax, palette="Set2")

    ax.set_xlabel("Rota")
    ax.set_ylabel("Fiyat (TL)")
    ax.set_title("Rota Bazlı Fiyat Dağılımı")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "02_price_by_route.png", dpi=150)
    plt.close()


def plot_price_by_airline(df: pd.DataFrame):
    """3. Havayolu bazlı fiyat karşılaştırması."""
    fig, ax = plt.subplots(figsize=(10, 6))

    sns.boxplot(data=df, x="airline", y="price", ax=ax, palette="Set3")
    ax.set_xlabel("Havayolu")
    ax.set_ylabel("Fiyat (TL)")
    ax.set_title("Havayolu Bazlı Fiyat Dağılımı")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "03_price_by_airline.png", dpi=150)
    plt.close()


def plot_bayram_effect(df: pd.DataFrame):
    """4. Bayram dönemi fiyat etkisi."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Bayram period boxplot
    period_order = ["Normal", "Pre-Bayram", "Bayram", "Post-Bayram"]
    sns.boxplot(data=df, x="bayram_period", y="price", order=period_order,
                ax=axes[0], palette=["#4ECDC4", "#FFD93D", "#FF6B6B", "#95E1D3"])
    axes[0].set_xlabel("Dönem")
    axes[0].set_ylabel("Fiyat (TL)")
    axes[0].set_title("Bayram Dönemine Göre Fiyatlar")

    # Günlük ortalama fiyat trendi
    daily_avg = df.groupby("depart_date")["price"].mean().reset_index()
    axes[1].plot(daily_avg["depart_date"], daily_avg["price"], marker=".", linewidth=1.5, color="#2E86C1")

    # Bayram bölgesini vurgula
    from datetime import date
    bayram_start = pd.Timestamp("2026-05-27")
    bayram_end = pd.Timestamp("2026-05-30")
    axes[1].axvspan(bayram_start, bayram_end, alpha=0.2, color="red", label="Bayram")
    axes[1].set_xlabel("Kalkış Tarihi")
    axes[1].set_ylabel("Ortalama Fiyat (TL)")
    axes[1].set_title("Günlük Ortalama Fiyat Trendi")
    axes[1].legend()
    axes[1].tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "04_bayram_effect.png", dpi=150)
    plt.close()


def plot_days_to_flight(df: pd.DataFrame):
    """5. Uçuşa kalan güne göre fiyat trendi."""
    fig, ax = plt.subplots(figsize=(12, 5))

    trend = df.groupby("days_to_flight")["price"].agg(["mean", "median"]).reset_index()
    ax.plot(trend["days_to_flight"], trend["median"], marker="o", markersize=4,
            linewidth=2, color="#2E86C1", label="Medyan Fiyat")
    ax.fill_between(trend["days_to_flight"],
                    trend["median"] * 0.9, trend["median"] * 1.1,
                    alpha=0.15, color="#2E86C1")

    ax.set_xlabel("Uçuşa Kalan Gün")
    ax.set_ylabel("Fiyat (TL)")
    ax.set_title("Booking Zamanlaması vs Fiyat (Ne kadar erken alınmalı?)")
    ax.invert_xaxis()
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "05_days_to_flight.png", dpi=150)
    plt.close()


def plot_time_and_weekday(df: pd.DataFrame):
    """6. Saat dilimi ve hafta günü etkisi."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Saat dilimi
    time_order = ["Sabah", "Öğle", "Akşam", "Gece"]
    time_df = df[df["time_of_day"].isin(time_order)]
    sns.boxplot(data=time_df, x="time_of_day", y="price", order=time_order,
                ax=axes[0], palette=["#FFD700", "#FF6347", "#4169E1", "#191970"])
    axes[0].set_xlabel("Saat Dilimi")
    axes[0].set_ylabel("Fiyat (TL)")
    axes[0].set_title("Kalkış Saatine Göre Fiyat")

    # Hafta günü
    weekday_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    weekday_avg = df.groupby("depart_weekday")["price"].median().reset_index()
    colors = ["#FF6347" if d in [5, 6] else "#4169E1" for d in weekday_avg["depart_weekday"]]
    axes[1].bar(weekday_avg["depart_weekday"], weekday_avg["price"], color=colors)
    axes[1].set_xticks(range(7))
    axes[1].set_xticklabels(weekday_names)
    axes[1].set_xlabel("Hafta Günü")
    axes[1].set_ylabel("Medyan Fiyat (TL)")
    axes[1].set_title("Hafta Gününe Göre Fiyat (Kırmızı=Hafta Sonu)")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "06_time_weekday.png", dpi=150)
    plt.close()


def plot_correlation_heatmap(df: pd.DataFrame):
    """7. Sayısal feature'ların korelasyon matrisi."""
    numeric_cols = ["price", "days_to_flight", "depart_weekday", "is_weekend",
                    "day_of_month", "depart_hour", "is_bayram", "days_to_bayram",
                    "route_avg_price", "airline_avg_price"]
    corr_cols = [c for c in numeric_cols if c in df.columns]
    corr = df[corr_cols].corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
                center=0, ax=ax, square=True, linewidths=0.5)
    ax.set_title("Feature Korelasyon Matrisi (Price ile İlişkiler)")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "07_correlation_heatmap.png", dpi=150)
    plt.close()

    return corr


def generate_summary(df: pd.DataFrame, corr: pd.DataFrame):
    """Özet istatistikleri txt olarak kaydet."""
    lines = []
    lines.append("=" * 60)
    lines.append("EDA ÖZET RAPORU")
    lines.append("=" * 60)

    lines.append(f"\n📊 Dataset: {len(df)} satır, {len(df.columns)} sütun")
    lines.append(f"📅 Tarih aralığı: {df['depart_date'].min().date()} → {df['depart_date'].max().date()}")
    lines.append(f"✈️  Rota sayısı: {df['route'].nunique()}")
    lines.append(f"🛫 Havayolu sayısı: {df['airline'].nunique()}")

    lines.append(f"\n{'─' * 60}")
    lines.append("FİYAT İSTATİSTİKLERİ")
    lines.append(f"{'─' * 60}")
    lines.append(f"  Min:      {df['price'].min():,.0f} TL")
    lines.append(f"  Q1:       {df['price'].quantile(0.25):,.0f} TL")
    lines.append(f"  Medyan:   {df['price'].median():,.0f} TL")
    lines.append(f"  Ortalama: {df['price'].mean():,.0f} TL")
    lines.append(f"  Q3:       {df['price'].quantile(0.75):,.0f} TL")
    lines.append(f"  Max:      {df['price'].max():,.0f} TL")
    lines.append(f"  Std:      {df['price'].std():,.0f} TL")

    lines.append(f"\n{'─' * 60}")
    lines.append("BAYRAM ETKİSİ")
    lines.append(f"{'─' * 60}")
    for period in ["Normal", "Pre-Bayram", "Bayram", "Post-Bayram"]:
        subset = df[df["bayram_period"] == period]
        if not subset.empty:
            lines.append(f"  {period:12s} → Medyan: {subset['price'].median():,.0f} TL | "
                         f"Ort: {subset['price'].mean():,.0f} TL | N={len(subset)}")

    normal_median = df[df["bayram_period"] == "Normal"]["price"].median()
    bayram_median = df[df["bayram_period"] == "Bayram"]["price"].median()
    if normal_median > 0:
        pct_increase = (bayram_median - normal_median) / normal_median * 100
        lines.append(f"\n  ⚠️  Bayram fiyat artışı: %{pct_increase:.1f} (medyan bazlı)")

    lines.append(f"\n{'─' * 60}")
    lines.append("PRICE İLE EN YÜKSEK KORELASYONLAR")
    lines.append(f"{'─' * 60}")
    price_corr = corr["price"].drop("price").sort_values(key=abs, ascending=False)
    for feat, val in price_corr.items():
        lines.append(f"  {feat:20s} → {val:+.3f}")

    lines.append(f"\n{'─' * 60}")
    lines.append("ROTA BAZLI MEDYAN FİYATLAR")
    lines.append(f"{'─' * 60}")
    route_medians = df.groupby("route")["price"].median().sort_values(ascending=False)
    for route, price in route_medians.items():
        lines.append(f"  {route:12s} → {price:,.0f} TL")

    lines.append(f"\n{'─' * 60}")
    lines.append("HAVAYOLU BAZLI MEDYAN FİYATLAR")
    lines.append(f"{'─' * 60}")
    airline_medians = df.groupby("airline")["price"].median().sort_values(ascending=False)
    for airline, price in airline_medians.items():
        lines.append(f"  {airline:12s} → {price:,.0f} TL")

    summary_text = "\n".join(lines)
    (OUTPUT_DIR / "00_eda_summary.txt").write_text(summary_text, encoding="utf-8")
    print(summary_text)


def main():
    print("📊 EDA başlıyor...\n")
    df = load_data()

    print("[1/7] Fiyat dağılımı...")
    plot_price_distribution(df)

    print("[2/7] Rota bazlı fiyatlar...")
    plot_price_by_route(df)

    print("[3/7] Havayolu bazlı fiyatlar...")
    plot_price_by_airline(df)

    print("[4/7] Bayram etkisi...")
    plot_bayram_effect(df)

    print("[5/7] Booking zamanlaması...")
    plot_days_to_flight(df)

    print("[6/7] Saat dilimi & hafta günü...")
    plot_time_and_weekday(df)

    print("[7/7] Korelasyon matrisi...")
    corr = plot_correlation_heatmap(df)

    print("\n" + "=" * 60)
    generate_summary(df, corr)

    print(f"\n\n✅ EDA tamamlandı! Grafikler: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
