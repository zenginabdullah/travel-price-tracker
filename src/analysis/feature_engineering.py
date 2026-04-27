from pathlib import Path
import pandas as pd
import numpy as np

INPUT_CSV = Path("data/processed/flights_enuygun_clean.csv")
OUTPUT_CSV = Path("data/processed/flights_features.csv")

def create_features(df):
    #Tarih dönüşümleri
    df["depart_date"] = pd.to_datetime(df["depart_date"], format="%Y-%m-%d")
    df["scrape_ts"] = pd.to_datetime(df["scrape_ts"]).dt.tz_convert("Europe/Istanbul")
    
    #Sadece tarihi alalım
    df["scrape_date"] = df["scrape_ts"].dt.date
    df["scrape_date"] = pd.to_datetime(df["scrape_date"])

    #Uçuşa Kalan Gün Sayısı
    df["days_to_flight"] = (df["depart_date"] - df["scrape_date"]).dt.days
    
    #Negatif günleri temizle
    df = df[df["days_to_flight"] >= 0].copy()

    #Kalkış Gününün Adı
    df["depart_weekday"] = df["depart_date"].dt.day_name()
    df["is_weekend"] = df["depart_weekday"].isin(["Saturday", "Sunday"]).astype(int)

    #Kalkış Saat Dilimi
    def get_time_of_day(time_str):
        try:
            hour = int(time_str.split(":")[0])
            if 5 <= hour < 12: return "Sabah"
            elif 12 <= hour < 17: return "Öğle"
            elif 17 <= hour < 22: return "Akşam"
            else: return "Gece"
        except:
            return "Bilinmiyor"

    df["time_of_day"] = df["depart_time"].apply(get_time_of_day)

    return df

def main():
    if not INPUT_CSV.exists():
        print(f"[HATA] {INPUT_CSV} bulunamadı. Önce temizleme scriptini çalıştırın.")
        return

    df = pd.read_csv(INPUT_CSV)
    df_features = create_features(df)
    
    df_features.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
    print(f"[OK] Özellikler üretildi. Toplam kayıt: {len(df_features)}")
    print(f"[OK] Kaydedildi: {OUTPUT_CSV}")

if __name__ == "__main__":
    main()