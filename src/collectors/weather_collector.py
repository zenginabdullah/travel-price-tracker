"""
Hava Durumu Veri Toplama (Open-Meteo API)
============================================
Projedeki şehirler için hava durumu verisini Open-Meteo Forecast API'den çeker.

Kullanılan API: Open-Meteo Forecast API (ücretsiz, API key gerekmez)
  - past_days=30 : 30 gün geriye dönük (geçmiş veri)
  - forecast_days=16 : 16 gün ileriye dönük (tahmin)
  - Toplam: 46 günlük veri

Doküman: https://open-meteo.com/en/docs

Çıktı:
  - data/processed/weather_data.csv : Tüm şehirler için günlük hava durumu
"""

import requests
import pandas as pd
from pathlib import Path
from datetime import date, timedelta
import time

# ─── Yollar ───────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
OUTPUT_CSV = PROJECT_ROOT / "data" / "processed" / "weather_data.csv"

# ─── Şehir Koordinatları ─────────────────────────────────────────────────────
# Projedeki 4 şehir: İstanbul, Antalya, Ankara, İzmir
CITIES = {
    "IST": {"name": "İstanbul", "lat": 41.01, "lon": 28.98},
    "AYT": {"name": "Antalya", "lat": 36.88, "lon": 30.70},
    "ESB": {"name": "Ankara", "lat": 39.92, "lon": 32.85},
    "ADB": {"name": "İzmir", "lat": 38.42, "lon": 27.14},
    "SAW": {"name": "İstanbul", "lat": 41.01, "lon": 28.98},  # SAW da İstanbul
}

# ─── Tarih Aralığı ───────────────────────────────────────────────────────────
# Projedeki uçuş verilerinin tarih aralığı
START_DATE = date(2026, 5, 1)
END_DATE = date(2026, 6, 14)

# ─── Hava Durumu Kodu Sınıflandırma ──────────────────────────────────────────
# Open-Meteo WMO weather codes
def classify_weather_condition(weather_code: int) -> str:
    """WMO kodunu kategorik hava durumuna çevir."""
    if weather_code in [0, 1, 2, 3]:
        return "Acik"
    elif weather_code in [45, 48]:
        return "Sisli"
    elif weather_code in [51, 53, 55, 56, 57]:
        return "Ciseleme"
    elif weather_code in [61, 63, 65, 66, 67, 80, 81, 82]:
        return "Yagmurlu"
    elif weather_code in [71, 73, 75, 77, 85, 86]:
        return "Karli"
    elif weather_code in [95, 96, 99]:
        return "Firtinali"
    else:
        return "Bilinmiyor"


def is_bad_weather(weather_code: int) -> int:
    """Kötü hava koşulu mu? (1=evet, 0=hayır)"""
    bad_codes = [45, 48, 65, 66, 67, 71, 73, 75, 77, 82, 85, 86, 95, 96, 99]
    return 1 if weather_code in bad_codes else 0


def fetch_weather_for_city(city_code: str, lat: float, lon: float) -> pd.DataFrame:
    """Bir şehir için Open-Meteo API'den hava durumu verisini çeker.
    
    Geçmiş veri (1 Mayıs - bugün) için Historical API,
    Tahmin (bugün - 14 Haziran) için Forecast API kullanılır.
    """
    
    today = date.today()
    
    # ─── Geçmiş veri: Historical API ────────────────────────────────────────
    past_end = today - timedelta(days=1)
    past_dfs = []
    
    if START_DATE <= past_end:
        hist_url = "https://archive-api.open-meteo.com/v1/archive"
        hist_params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": str(max(START_DATE, date(2026, 4, 1))),
            "end_date": str(past_end),
            "daily": ["temperature_2m_max", "temperature_2m_min", 
                      "precipitation_sum", "wind_speed_10m_max", "weather_code"],
            "timezone": "Europe/Istanbul",
        }
        print(f"  [API-Hist] {city_code} -> {hist_params['start_date']} -> {hist_params['end_date']}")
        
        try:
            resp = requests.get(hist_url, params=hist_params, timeout=30)
            resp.raise_for_status()
            hist_data = resp.json()
            hist_daily = hist_data.get("daily", {})
            if hist_daily and "time" in hist_daily:
                past_dfs.append(pd.DataFrame({
                    "date": hist_daily["time"],
                    "city_code": city_code,
                    "temp_max": hist_daily["temperature_2m_max"],
                    "temp_min": hist_daily["temperature_2m_min"],
                    "precipitation": hist_daily["precipitation_sum"],
                    "wind_speed": hist_daily["wind_speed_10m_max"],
                    "weather_code": hist_daily["weather_code"],
                }))
                print(f"    [OK] {len(hist_daily['time'])} gun (gecmis)")
        except Exception as e:
            print(f"    [WARN] Historical API hatasi: {e}")
    
    # ─── Tahmin: Forecast API ───────────────────────────────────────────────
    if today <= END_DATE:
        fc_url = "https://api.open-meteo.com/v1/forecast"
        fc_params = {
            "latitude": lat,
            "longitude": lon,
            "daily": ["temperature_2m_max", "temperature_2m_min",
                      "precipitation_sum", "wind_speed_10m_max", "weather_code"],
            "timezone": "Europe/Istanbul",
            "forecast_days": min(16, (END_DATE - today).days + 1),
        }
        print(f"  [API-Fcst] {city_code} -> forecast_days={fc_params['forecast_days']}")
        
        try:
            resp = requests.get(fc_url, params=fc_params, timeout=30)
            resp.raise_for_status()
            fc_data = resp.json()
            fc_daily = fc_data.get("daily", {})
            if fc_daily and "time" in fc_daily:
                past_dfs.append(pd.DataFrame({
                    "date": fc_daily["time"],
                    "city_code": city_code,
                    "temp_max": fc_daily["temperature_2m_max"],
                    "temp_min": fc_daily["temperature_2m_min"],
                    "precipitation": fc_daily["precipitation_sum"],
                    "wind_speed": fc_daily["wind_speed_10m_max"],
                    "weather_code": fc_daily["weather_code"],
                }))
                print(f"    [OK] {len(fc_daily['time'])} gun (tahmin)")
        except Exception as e:
            print(f"    [WARN] Forecast API hatasi: {e}")
    
    if not past_dfs:
        print(f"  [WARN] {city_code} icin veri alinamadi")
        return pd.DataFrame()
    
    df = pd.concat(past_dfs, ignore_index=True)
    
    # NaN değerleri doldur
    df["temp_max"] = df["temp_max"].fillna(df["temp_max"].median())
    df["temp_min"] = df["temp_min"].fillna(df["temp_min"].median())
    df["precipitation"] = df["precipitation"].fillna(0)
    df["wind_speed"] = df["wind_speed"].fillna(df["wind_speed"].median())
    df["weather_code"] = df["weather_code"].fillna(0).astype(int)
    
    # Kategorik sütunlar
    df["weather_condition"] = df["weather_code"].apply(classify_weather_condition)
    df["is_bad_weather"] = df["weather_code"].apply(is_bad_weather)
    
    # Sadece proje tarih aralığı
    df["date_parsed"] = pd.to_datetime(df["date"]).dt.date
    df = df[(df["date_parsed"] >= START_DATE) & (df["date_parsed"] <= END_DATE)]
    df = df.drop(columns=["date_parsed"])
    
    print(f"  [OK] {city_code}: {len(df)} gun (toplam)")
    return df


def main():
    print("=" * 60)
    print("  HAVA DURUMU VERI TOPLAMA")
    print("=" * 60)
    print(f"\nProje tarih araligi: {START_DATE} -> {END_DATE}")
    print(f"Sehirler: {list(CITIES.keys())}")
    print(f"API: past_days=30 + forecast_days=16 = 46 gunluk veri")
    
    all_dfs = []
    
    for city_code, info in CITIES.items():
        print(f"\n{'─' * 40}")
        print(f"[{city_code}] {info['name']} verisi cekiliyor...")
        
        df = fetch_weather_for_city(
            city_code=city_code,
            lat=info["lat"],
            lon=info["lon"],
        )
        
        if not df.empty:
            all_dfs.append(df)
        
        # Rate limiting - API'yi yormamak için bekle
        time.sleep(1.0)
    
    if not all_dfs:
        print("\n[ERROR] Hicbir sehir icin veri alinamadi!")
        return
    
    # Tüm şehirleri birleştir
    weather_df = pd.concat(all_dfs, ignore_index=True)
    
    # Sırala
    weather_df = weather_df.sort_values(["city_code", "date"]).reset_index(drop=True)
    
    # Kaydet
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    weather_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
    
    print(f"\n{'=' * 60}")
    print(f"[OK] Hava durumu verisi kaydedildi!")
    print(f"     Dosya: {OUTPUT_CSV}")
    print(f"     Kayit: {len(weather_df)} (sehir x gun)")
    print(f"     Sutunlar: {list(weather_df.columns)}")
    print(f"\nOzet (proje araligindaki veriler):")
    print(weather_df.groupby("city_code").agg({
        "temp_max": "mean",
        "precipitation": "sum",
        "wind_speed": "mean",
    }).round(1).to_string())


if __name__ == "__main__":
    main()
