"""
Feature Engineering - ML'e Hazır Dataset Üretimi (Aggregated)
==============================================================
Her (rota, kalkış_tarihi) kombinasyonu için tek satır üretir.
Hedef: O gün o rotanın medyan fiyatı.

Yeni: Hava durumu feature'ları eklendi (Open-Meteo API'den alınan veri).
  - arrival_temp_max      : Varış şehrindeki maksimum sıcaklık
  - arrival_precipitation : Varış şehrindeki yağış miktarı
  - arrival_wind_speed    : Varış şehrindeki rüzgar hızı
  - arrival_weather_condition : Kategorik hava durumu (Açık/Yağmurlu vb.)
  - is_bad_weather        : Kötü hava koşulu mu? (binary)

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
WEATHER_CSV = PROJECT_ROOT / "data" / "processed" / "weather_data.csv"
OUTPUT_FEATURES = PROJECT_ROOT / "data" / "processed" / "flights_features.csv"
OUTPUT_ML = PROJECT_ROOT / "data" / "processed" / "flights_ml_ready.csv"

# ─── Sabitler ─────────────────────────────────────────────────────────────────
BAYRAM_START = date(2026, 5, 27)
BAYRAM_END = date(2026, 5, 30)
PRE_BAYRAM_DAYS = 7
POST_BAYRAM_DAYS = 7

# ─── Şehir Kodları (varış şehrine göre hava durumu eşleme) ────────────────────
# Her rotanın varış noktasına göre hangi şehir kodunu kullanacağımız
DESTINATION_CITY_MAP = {
    "IST": "IST",
    "AYT": "AYT",
    "ESB": "ESB",
    "ADB": "ADB",
    "SAW": "IST",  # SAW da İstanbul
}


def load_weather_data() -> pd.DataFrame:
    """Hava durumu verisini yükle ve tarih formatını düzenle."""
    if not WEATHER_CSV.exists():
        print(f"[WARN] Hava durumu verisi bulunamadı: {WEATHER_CSV}")
        print("       Önce weather_collector.py çalıştırılmalı.")
        return pd.DataFrame()
    
    weather = pd.read_csv(WEATHER_CSV)
    weather["date"] = pd.to_datetime(weather["date"])
    return weather


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
            return "Bayram Oncesi"
        elif BAYRAM_END < d <= (BAYRAM_END + timedelta(days=POST_BAYRAM_DAYS)):
            return "Bayram Sonrasi"
        else:
            return "Normal"

    df["bayram_period"] = depart_dates.apply(classify_bayram_period)

    # Rota
    df["route"] = df["origin"] + "_" + df["destination"]

    # ─── Hava Durumu Feature'ları ─────────────────────────────────────────────
    weather = load_weather_data()
    if not weather.empty:
        # Varış şehrine göre city_code belirle
        df["dest_city_code"] = df["destination"].map(DESTINATION_CITY_MAP)
        
        # Merge: depart_date + dest_city_code ile weather verisini birleştir
        df = df.merge(
            weather[["date", "city_code", "temp_max", "temp_min", "precipitation",
                     "wind_speed", "weather_condition", "is_bad_weather"]],
            left_on=["depart_date", "dest_city_code"],
            right_on=["date", "city_code"],
            how="left"
        )
        
        # Merge'den gelen gereksiz sütunları temizle
        df = df.drop(columns=["date", "city_code", "dest_city_code"])
        
        # NaN olan hava durumu değerlerini doldur (eşleşmeyen günler için)
        df["temp_max"] = df["temp_max"].fillna(df.groupby("destination")["temp_max"].transform("median"))
        df["temp_min"] = df["temp_min"].fillna(df.groupby("destination")["temp_min"].transform("median"))
        df["precipitation"] = df["precipitation"].fillna(0)
        df["wind_speed"] = df["wind_speed"].fillna(df.groupby("destination")["wind_speed"].transform("median"))
        # Eşleşmeyen günler için en yaygın hava durumunu kullan
        most_common_weather = df["weather_condition"].mode()
        df["weather_condition"] = df["weather_condition"].fillna(most_common_weather.iloc[0] if not most_common_weather.empty else "Acik")
        df["is_bad_weather"] = df["is_bad_weather"].fillna(0).astype(int)
        
        # Sıcaklık farkı (kalkış - varış arası sıcaklık farkı)
        # İstanbul için kalkış sıcaklığını tahmin etmek zor, o yüzden sadece varış sıcaklığını kullanıyoruz
        
        print(f"      Hava durumu eklendi: {weather['city_code'].nunique()} sehir")
    else:
        print("      [WARN] Hava durumu verisi olmadan devam ediliyor...")
        df["temp_max"] = 20.0
        df["temp_min"] = 10.0
        df["precipitation"] = 0.0
        df["wind_speed"] = 10.0
        df["weather_condition"] = "Bilinmiyor"
        df["is_bad_weather"] = 0

    return df


def create_aggregated_ml_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Her (rota, kalkış_tarihi) için aggregated satır üretir.
    
    Hedef: median_price
    Feature'lar: Zaman, bayram, rota bilgileri + fiyat istatistikleri + hava durumu
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
        # Hava durumu (hepsi aynı gün için aynı)
        temp_max=("temp_max", "first"),
        temp_min=("temp_min", "first"),
        precipitation=("precipitation", "first"),
        wind_speed=("wind_speed", "first"),
        weather_condition=("weather_condition", "first"),
        is_bad_weather=("is_bad_weather", "first"),
    ).reset_index()

    # Std NaN olabilir (tek uçuş varsa)
    agg_df["std_price"] = agg_df["std_price"].fillna(0)

    # Fiyat aralığı
    agg_df["price_spread"] = agg_df["max_price"] - agg_df["min_price"]
    agg_df["price_iqr"] = agg_df["q75_price"] - agg_df["q25_price"]

    # ─── Rota bazlı global istatistikler ─────────────────────────────────────
    route_global = agg_df.groupby("route")["min_price"].agg(["mean", "std"]).rename(
        columns={"mean": "route_global_avg", "std": "route_global_std"}
    )
    agg_df = agg_df.merge(route_global, on="route", how="left")

    # ─── Hafta numarası (ay içi hafta) ───────────────────────────────────────
    agg_df["week_of_month"] = (agg_df["day_of_month"] - 1) // 7 + 1

    # ─── Bayram'a yakınlık (non-linear) ──────────────────────────────────────
    agg_df["bayram_proximity"] = np.exp(-agg_df["days_to_bayram"] / 7)

    # ─── ML-ready format ─────────────────────────────────────────────────────
    ml_features = [
        # Hedef
        "min_price",
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
        # Hava durumu
        "temp_max",
        "temp_min",
        "precipitation",
        "wind_speed",
        "weather_condition",
        "is_bad_weather",
    ]

    ml_df = agg_df[ml_features].copy()

    # One-hot encode
    ml_df = pd.get_dummies(
        ml_df,
        columns=["route", "bayram_period", "weather_condition"],
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
    print(f"     Hedef değişken: min_price")
    print(f"     Feature sayısı: {len(df_ml.columns) - 1}")
    print(f"     Min fiyat aralığı: {df_ml['min_price'].min():.0f} - {df_ml['min_price'].max():.0f} TL")
    
    # Hava durumu feature'larının varlığını kontrol et
    weather_cols = [c for c in df_ml.columns if any(w in c for w in 
                    ["temp_", "precipitation", "wind_speed", "weather_", "is_bad"])]
    if weather_cols:
        print(f"     Hava durumu feature'lari: {len(weather_cols)} adet")
        print(f"       → {weather_cols}")


if __name__ == "__main__":
    main()
