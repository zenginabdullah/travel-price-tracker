# travel-price-tracker
Location-based hotel and flight price tracking for data mining coursework.

# 🌍 Travel Price Tracker

> Bu proje, **Veri Madenciliği** dersi kapsamında belirli bir rota ve tarih aralığı için otel ve uçak bileti fiyatlarının zaman içindeki değişimini takip etmeyi amaçlar.

**Not:** Bu README proje başlangıç sürümüdür. Çıktı grafikleri ve sonuçlar veri toplama süreci ilerledikçe eklenecektir.

---

## 🎯 Projenin Temel Hedefi
* Düzenli aralıklarla fiyat verisi toplamak.
* Verileri ortak bir şemada saklamak.
* Zaman serisi mantığıyla fiyat değişimlerini analiz etmek.
* Basit alarm/işaretleme kuralları oluşturmak (ör. fiyat düşüşü).

## 🔭 Proje Kapsamı (Başlangıç)
Başlangıç aşamasında kapsam bilinçli olarak dar tutulmuştur. Amaç; önce çalışır bir MVP oluşturmak, sonra kapsamı genişletmektir.
* **Uçuş:** 1-2 sabit rota (ör. İstanbul → Antalya)
* **Otel:** 1 sabit bölge (ör. merkez nokta + yarıçap yaklaşımı)
* **Zamanlama:** Günde 1 kez veya sınırlı periyotlarda veri toplama

Uçuş verisi toplama tarafında ham çıktılar artık rota ve gün bazında ayrı JSON dosyaları olarak `data/raw/` altında saklanır. Örnek adlandırma: `flights_enuygun_20260501_IST_AYT.json`.

## ⚖️ Etik ve Hukuki Çerçeve
Web kazıma (web scraping) adımlarına geçmeden önce aşağıdaki ilkelere uyulur:
* Hedef kaynakların kullanım koşulları (ToS) ve `robots.txt` kontrol edilir.
* Kişisel veri toplanmaz (isim, e-posta vb. yok).
* Sadece kamusal teklif/fiyat bilgileri işlenir.
* Düşük frekanslı ve saygılı istek politikası uygulanır (rate limiting).
* Gerekirse manuel örnekleme ve aynı analiz hattı ile devam edilir.

---

## 💾 Planlanan Veri Modeli
Projede iki temel veri grubu bulunur:

### 1. Otel Fiyat Kayıtları
| Alan | Açıklama |
| :--- | :--- |
| `scrape_ts` | Kazıma zaman damgası |
| `source` | Veri kaynağı (Platform adı vb.) |
| `hotel_name` | Otel adı |
| `lat`, `lng` | Koordinatlar veya adres |
| `check_in`, `check_out` | Giriş ve çıkış tarihleri |
| `price`, `currency` | Fiyat ve para birimi |
| `rating` | Otel puanı (varsa) |
| `raw_url` | İlanın ham linki |
| `run_id` | Çalıştırma ID'si (Log takibi için) |

### 2. Uçak Fiyat Kayıtları
| Alan | Açıklama |
| :--- | :--- |
| `scrape_ts` | Kazıma zaman damgası |
| `source` | Veri kaynağı |
| `origin`, `destination` | Kalkış ve varış noktası |
| `depart_date`, `return_date` | Gidiş ve dönüş tarihi (varsa) |
| `airline` | Havayolu şirketi (varsa) |
| `price`, `currency` | Fiyat ve para birimi |
| `raw_url` | İlanın ham linki |
| `run_id` | Çalıştırma ID'si |

---

## ⚙️ Mimari Akış (Plan)
1. **Zamanlanmış Çalıştırma:** Scheduler (Cron vb.)
2. **Veri Toplama:** Collector modülü
3. **Ham Veri Kaydı:** `data/raw` içerisine JSON/CSV olarak yedekleme
4. **Temizleme ve Doğrulama:** `src/clean`
5. **Veritabanına Yazma:** `src/db`
6. **Analiz:** `src/analysis` üzerinden zaman serisi işlemleri
7. **Raporlama ve Görselleştirme:** `reports` (Grafikler, metrikler)

---

## 📂 Klasör Yapısı
```text
travel-price-tracker/
├─ src/
│  ├─ collectors/     # Web scraping scriptleri
│  ├─ clean/          # Veri temizleme ve normalizasyon
│  ├─ db/             # Veritabanı bağlantı ve kayıt işlemleri
│  └─ analysis/       # Fiyat analizleri ve alarmlar
├─ data/
│  ├─ raw/            # Ham kazınmış veri (JSON/CSV)
│  └─ processed/      # Temizlenmiş ve işlenmiş veri
├─ reports/           # Çıktı grafikleri ve markdown raporları
├─ .env.example       # Örnek çevre değişkenleri dosyası
├─ requirements.txt   # Python bağımlılıkları
└─ README.md
```
---
## Lisans
Bu depo ders/akademik çalışma amaçlı hazırlanmıştır.
