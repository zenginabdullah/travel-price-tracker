"""
Feature Engineering - ML'e Hazır Dataset Üretimi
=================================================
Temizlenmiş CSV'yi alır, model eğitimi için gerekli tüm özellikleri türetir.

Çıktı olarak iki dosya üretir:
  - flights_features.csv : Tüm feature'lar dahil (EDA ve öneri sistemi için)
  - flights_ml_ready.csv : Sadece ML modeline girecek sütunlar (leakage-free)
"""

from pathlib import Path
import pandas as pd
import numpy as np
from datetime import date

# ─── Yollar ───────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
INPUT_CSV = PROJECT_ROOT / "data" / "processed" / "flights_enuygun_clean.csv"
OUTPUT_FEATURES = PROJECT_ROOT / "data" / "processed" / "flights_features.csv"
OUTPUT_ML = PROJECT_ROOT / "data" / "processed" / "flights_ml_ready.csv"

# ─── Sabitler ─────────────────────────────────────────────────────────────────
BAYRAM_START = date(2026, 5, 27)
BAYRAM_END = date(2026, 5, 30)
PRE_BAYRAM_DAYS = 7
POST_BAYRAM_DAYS = 7


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """Tüm feature'ları türetir."""

    # ─── Tarih Dönüşümleri ────────────────────────────────────────────────────
    df["depart_date"] = pd.to_datetime(df["depart_date"], format="%Y-%m-%d")
    df["scrape_ts"] = pd.to_datetime(df["scrape_ts"]).dt.tz_convert("Europe/Istanbul")
    df["scrape_date"] = df["scrape_ts"].dt.normalize().dt.tz_localize(None)

    # ─── Temel Zaman Feature'ları ─────────────────────────────────────────────
    df["days_to_flight"] = (df["depart_date"] - df["scrape_date"]).dt.days
    df = df[df["days_to_flight"] >= 0].copy()

    df["depart_weekday"] = df["depart_date"].dt.dayofweek          # 0=Mon, 6=Sun
    df["depart_weekday_name"] = df["depart_date"].dt.day_name()
    df["is_weekend"] = df["depart_weekday"].isin([5, 6]).astype(int)
    df["day_of_month"] = df["depart_date"].dt.day

    # ─── Kalkış Saati (Sayısal) ──────────────────────────────────────────────
    def extract_hour(time_str):
        try:
            return int(str(time_str).split(":")[0])
        except (ValueError, AttributeError):
            return -1

    df["depart_hour"] = df["depart_time"].apply(extract_hour)

    # Saat dilimi kategorisi
    def get_time_of_day(hour):
        if hour < 0:
            return "Bilinmiyor"
        if 5 <= hour < 12:
            return "Sabah"
        elif 12 <= hour < 17:
            return "Öğle"
        elif 17 <= hour < 22:
            return "Akşam"
        else:
            return "Gece"

    df["time_of_day"] = df["depart_hour"].apply(get_time_of_day)

    # ─── Bayram Feature'ları ──────────────────────────────────────────────────
    depart_dates = df["depart_date"].dt.date

    df["is_bayram"] = depart_dates.apply(
        lambda d: 1 if BAYRAM_START <= d <= BAYRAM_END else 0
    )

    df["days_to_bayram"] = depart_dates.apply(
        lambda d: (BAYRAM_START - d).days if d < BAYRAM_START
        else (d - BAYRAM_END).days if d > BAYRAM_END
        else 0
    )

    def classify_bayram_period(d):
        from datetime import timedelta
        if BAYRAM_START <= d <= BAYRAM_END:
            return "Bayram"
        elif (BAYRAM_START - timedelta(days=PRE_BAYRAM_DAYS)) <= d < BAYRAM_START:
            return "Pre-Bayram"
        elif BAYRAM_END < d <= (BAYRAM_END + timedelta(days=POST_BAYRAM_DAYS)):
            return "Post-Bayram"
        else:
            return "Normal"

    df["bayram_period"] = depart_dates.apply(classify_bayram_period)

    # ─── Rota & Havayolu Feature'ları ─────────────────────────────────────────
    df["route"] = df["origin"] + "_" + df["destination"]

    # Rota bazlı ortalama fiyat (global referans)
    route_stats = df.groupby("route")["price"].agg(["mean", "median"]).rename(
        columns={"mean": "route_avg_price", "median": "route_median_price"}
    )
    df = df.merge(route_stats, on="route", how="left")

    # Havayolu bazlı ortalama fiyat
    airline_stats = df.groupby("airline")["price"].agg(["mean", "median"]).rename(
        columns={"mean": "airline_avg_price", "median": "airline_median_price"}
    )
    df = df.merge(airline_stats, on="airline", how="left")

    # Fiyatın rota ortalamasından sapması (%)
    df["price_vs_route_avg"] = ((df["price"] - df["route_avg_price"]) / df["route_avg_price"] * 100).round(2)

    # ─── Fiyat Yüzdelik & Anomali (Sadece EDA/Öneri için, ML'de kullanılmaz) ─
    df["price_percentile"] = df.groupby(
        ["route", "depart_weekday", "time_of_day"],
        group_keys=False
    )["price"].transform(lambda x: x.rank(pct=True) * 100)

    df["is_price_outlier"] = df.groupby(
        ["route", "depart_date"],
        group_keys=False
    )["price"].transform(
        lambda x: (x > (x.quantile(0.75) + 1.5 * (x.quantile(0.75) - x.quantile(0.25)))).astype(int)
    )

    return df


def prepare_ml_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    ML modeli için leakage-free dataset hazırlar.
    
    Hedef değişken: price
    Çıkarılanlar: price'dan türetilmiş feature'lar, URL, timestamp'ler, raw text'ler
    """

    # ML'de kullanılacak feature sütunları
    ml_features = [
        # Hedef
        "price",
        # Zaman
        "days_to_flight",
        "depart_weekday",
        "is_weekend",
        "day_of_month",
        "depart_hour",
        # Bayram
        "is_bayram",
        "days_to_bayram",
        "bayram_period",
        # Rota & Havayolu (kategorik - encode edilecek)
        "route",
        "airline",
        "time_of_day",
        # Referans istatistikler (global ortalamalar - leakage değil çünkü tüm veriden)
        "route_avg_price",
        "airline_avg_price",
    ]

    ml_df = df[ml_features].copy()

    # Bilinmeyen saat dilimlerini çıkar
    ml_df = ml_df[ml_df["time_of_day"] != "Bilinmiyor"]
    ml_df = ml_df[ml_df["depart_hour"] >= 0]

    # Kategorik sütunları one-hot encode et
    ml_df = pd.get_dummies(
        ml_df,
        columns=["route", "airline", "time_of_day", "bayram_period"],
        drop_first=False,
        dtype=int
    )

    return ml_df


def main():
    if not INPUT_CSV.exists():
        print(f"[HATA] {INPUT_CSV} bulunamadı. Önce temizleme scriptini çalıştırın.")
        return

    print("[1/3] Veri okunuyor...")
    df = pd.read_csv(INPUT_CSV)
    print(f"      Ham kayıt: {len(df)}")

    print("[2/3] Feature'lar türetiliyor...")
    df_features = create_features(df)
    df_features.to_csv(OUTPUT_FEATURES, index=False, encoding="utf-8")
    print(f"      Feature kayıt: {len(df_features)}")
    print(f"      → {OUTPUT_FEATURES}")

    print("[3/3] ML-ready dataset hazırlanıyor...")
    df_ml = prepare_ml_dataset(df_features)
    df_ml.to_csv(OUTPUT_ML, index=False, encoding="utf-8")
    print(f"      ML kayıt: {len(df_ml)}")
    print(f"      ML sütunları: {len(df_ml.columns)}")
    print(f"      → {OUTPUT_ML}")

    print("\n[OK] Tamamlandı!")
    print(f"     Toplam feature sayısı (encoded): {len(df_ml.columns) - 1} + hedef (price)")


if __name__ == "__main__":
    main()
