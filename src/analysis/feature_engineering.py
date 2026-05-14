"""
Feature Engineering - ML'e Hazır Dataset Üretimi (Aggregated)
==============================================================
Her (rota, kalkış_tarihi) kombinasyonu için tek satır üretir.
Hedef: O gün o rotanın medyan fiyatı.

Bu sayede:
  - Aynı uçuşun farklı fare class gürültüsü elenir
  - Model tutarlı pattern'ları öğrenir
  - Kullanıcıya "bu rota bu gün normalde şu fiyat" bilgisi verilir

Çıktılar:
  - flights_features.csv     : Tüm detaylı veri (EDA için)
  - flights_ml_ready.csv     : Aggregated, ML modeline girecek dataset
"""

from pathlib import Path
import pandas as pd
import numpy as np
from datetime import date, timedelta

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


def create_base_features(df: pd.DataFrame) -> pd.DataFrame:
    """Ham veriye temel feature'ları ekler (detaylı EDA için)."""

    # Tarih dönüşümleri
    df["depart_date"] = pd.to_datetime(df["depart_date"], format="%Y-%m-%d")
    df["scrape_ts"] = pd.to_datetime(df["scrape_ts"]).dt.tz_convert("Europe/Istanbul")
    df["scrape_date"] = df["scrape_ts"].dt.normalize().dt.tz_localize(None)

    # Temel zaman
    df["days_to_flight"] = (df["depart_date"] - df["scrape_date"]).dt.days
    df = df[df["days_to_flight"] >= 0].copy()

    df["depart_weekday"] = df["depart_date"].dt.dayofweek
    df["depart_weekday_name"] = df["depart_date"].dt.day_name()
    df["is_weekend"] = df["depart_weekday"].isin([5, 6]).astype(int)
    df["day_of_month"] = df["depart_date"].dt.day

    # Kalkış saati
    def extract_hour(time_str):
        try:
            return int(str(time_str).split(":")[0])
        except (ValueError, AttributeError):
            return -1

    df["depart_hour"] = df["depart_time"].apply(extract_hour)

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

    # Bayram
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
        if BAYRAM_START <= d <= BAYRAM_END:
            return "Bayram"
        elif (BAYRAM_START - timedelta(days=PRE_BAYRAM_DAYS)) <= d < BAYRAM_START:
            return "Pre-Bayram"
        elif BAYRAM_END < d <= (BAYRAM_END + timedelta(days=POST_BAYRAM_DAYS)):
            return "Post-Bayram"
        else:
            return "Normal"

    df["bayram_period"] = depart_dates.apply(classify_bayram_period)

    # Rota
    df["route"] = df["origin"] + "_" + df["destination"]

    return df


def create_aggregated_ml_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Her (rota, kalkış_tarihi) için aggregated satır üretir.
    
    Hedef: median_price
    Feature'lar: Zaman, bayram, rota bilgileri + fiyat istatistikleri
    """

    # ─── Aggregation: rota + gün bazında ─────────────────────────────────────
    agg_df = df.groupby(["route", "depart_date"]).agg(
        median_price=("price", "median"),
        min_price=("price", "min"),
        max_price=("price", "max"),
        mean_price=("price", "mean"),
        std_price=("price", "std"),
        q25_price=("price", lambda x: x.quantile(0.25)),
        q75_price=("price", lambda x: x.quantile(0.75)),
        flight_count=("price", "count"),
        airline_count=("airline", "nunique"),
        # İlk satırdan zaman bilgilerini al (hepsi aynı gün)
        days_to_flight=("days_to_flight", "first"),
        depart_weekday=("depart_weekday", "first"),
        is_weekend=("is_weekend", "first"),
        day_of_month=("day_of_month", "first"),
        is_bayram=("is_bayram", "first"),
        days_to_bayram=("days_to_bayram", "first"),
        bayram_period=("bayram_period", "first"),
    ).reset_index()

    # Std NaN olabilir (tek uçuş varsa)
    agg_df["std_price"] = agg_df["std_price"].fillna(0)

    # Fiyat aralığı
    agg_df["price_spread"] = agg_df["max_price"] - agg_df["min_price"]
    agg_df["price_iqr"] = agg_df["q75_price"] - agg_df["q25_price"]

    # ─── Rota bazlı global istatistikler ─────────────────────────────────────
    route_global = agg_df.groupby("route")["median_price"].agg(["mean", "std"]).rename(
        columns={"mean": "route_global_avg", "std": "route_global_std"}
    )
    agg_df = agg_df.merge(route_global, on="route", how="left")

    # ─── Hafta numarası (ay içi hafta) ───────────────────────────────────────
    agg_df["week_of_month"] = (agg_df["day_of_month"] - 1) // 7 + 1

    # ─── Bayram'a yakınlık (non-linear) ──────────────────────────────────────
    # Bayrama yaklaştıkça fiyat artışı üstel olabilir
    agg_df["bayram_proximity"] = np.exp(-agg_df["days_to_bayram"] / 7)

    # ─── ML-ready format ─────────────────────────────────────────────────────
    ml_features = [
        # Hedef
        "median_price",
        # Zaman
        "days_to_flight",
        "depart_weekday",
        "is_weekend",
        "day_of_month",
        "week_of_month",
        # Bayram
        "is_bayram",
        "days_to_bayram",
        "bayram_proximity",
        "bayram_period",
        # Rota (kategorik)
        "route",
        # Arz bilgisi (o gün kaç uçuş/havayolu var)
        "flight_count",
        "airline_count",
        # Rota global referans
        "route_global_avg",
        "route_global_std",
    ]

    ml_df = agg_df[ml_features].copy()

    # One-hot encode
    ml_df = pd.get_dummies(
        ml_df,
        columns=["route", "bayram_period"],
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

    print("[2/3] Temel feature'lar türetiliyor...")
    df_features = create_base_features(df)
    df_features.to_csv(OUTPUT_FEATURES, index=False, encoding="utf-8")
    print(f"      Feature kayıt: {len(df_features)}")
    print(f"      → {OUTPUT_FEATURES}")

    print("[3/3] Aggregated ML dataset hazırlanıyor...")
    df_ml = create_aggregated_ml_dataset(df_features)
    df_ml.to_csv(OUTPUT_ML, index=False, encoding="utf-8")
    print(f"      ML kayıt: {len(df_ml)} (rota×gün kombinasyonu)")
    print(f"      ML sütunları: {len(df_ml.columns)}")
    print(f"      → {OUTPUT_ML}")

    # Özet
    print(f"\n[OK] Tamamlandı!")
    print(f"     Hedef değişken: median_price")
    print(f"     Feature sayısı: {len(df_ml.columns) - 1}")
    print(f"     Medyan fiyat aralığı: {df_ml['median_price'].min():.0f} - {df_ml['median_price'].max():.0f} TL")


if __name__ == "__main__":
    main()
