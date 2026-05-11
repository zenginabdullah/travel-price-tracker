# ✈️ Travel Price Tracker

> **Veri Madenciliği** dersi kapsamında, belirli rotalar ve tarih aralığı için uçak bileti fiyatlarının zaman içindeki değişimini takip eden ve ML tabanlı öneri sistemi geliştirmeyi hedefleyen bir projedir.

---

## 🎯 Projenin Hedefi

* Enuygun.com'dan düzenli aralıklarla uçuş fiyat verisi toplamak
* Verileri temizleyip ortak bir şemada saklamak
* Özellik mühendisliği ile ML'e hazır dataset oluşturmak
* Fiyat tahmini modeli eğitip "şimdi al / bekle" önerisi sunmak

---

## 📊 Veri Kapsamı

| Parametre | Değer |
|-----------|-------|
| **Kaynak** | Enuygun.com |
| **Tarih Aralığı** | 01.05.2026 – 14.06.2026 |
| **Rotalar** | IST↔AYT, SAW↔ESB, IST↔ADB (6 yön) |
| **Toplam Kayıt** | ~8.707 uçuş |
| **Özel Dönem** | Kurban Bayramı (27-30 Mayıs 2026) |

---

## ⚙️ Pipeline

```
1. Veri Kazıma (Playwright)  →  data/raw/{YYYYMMDD}/{ORIGIN}_{DEST}.json
2. Temizleme & Doğrulama     →  data/processed/flights_enuygun_clean.csv
3. Özellik Mühendisliği      →  data/processed/flights_features.csv
4. ML Model (TODO)           →  Fiyat tahmini & öneri
```

---

## 📂 Klasör Yapısı

```text
travel-price-tracker/
├─ src/
│  ├─ collectors/
│  │  └─ flight_scraper_enuygun.py   # Playwright ile veri kazıma
│  ├─ clean/
│  │  └─ clean_flights_enuygun.py    # Ham JSON → temiz CSV
│  └─ analysis/
│     └─ feature_engineering.py      # Türetilmiş özellikler
├─ data/
│  ├─ raw/            # Ham JSON (gün/rota bazlı)
│  └─ processed/      # Temizlenmiş CSV'ler
├─ requirements.txt
└─ README.md
```

---

## 🚀 Kurulum & Çalıştırma

```bash
# Sanal ortam
python -m venv .venv
.venv\Scripts\activate

# Bağımlılıklar
pip install -r requirements.txt
playwright install chromium

# Pipeline adımları
python src/collectors/flight_scraper_enuygun.py   # Veri kazıma
python src/clean/clean_flights_enuygun.py         # Temizleme
python src/analysis/feature_engineering.py        # Özellik üretimi
```

---

## ⚖️ Etik ve Hukuki Çerçeve

* Hedef kaynakların kullanım koşulları ve `robots.txt` kontrol edilmiştir
* Kişisel veri toplanmaz
* Sadece kamusal fiyat bilgileri işlenir
* Düşük frekanslı ve saygılı istek politikası uygulanır (rate limiting + jitter)

---

## 📋 Türetilmiş Özellikler

| Özellik | Açıklama |
|---------|----------|
| `days_to_flight` | Uçuşa kalan gün sayısı |
| `depart_weekday` | Kalkış günü adı |
| `is_weekend` | Hafta sonu mu? |
| `time_of_day` | Saat dilimi (Sabah/Öğle/Akşam/Gece) |
| `is_bayram` | Bayram günü mü? |
| `days_to_bayram` | Bayrama kalan gün |
| `bayram_period` | Normal / Pre-Bayram / Bayram / Post-Bayram |
| `price_percentile` | Rota+saat kombinasyonuna göre fiyat yüzdeliği |
| `is_price_outlier` | IQR tabanlı anomali flag'i |

---

## Lisans

Bu depo ders/akademik çalışma amaçlı hazırlanmıştır.
