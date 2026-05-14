"""
✈️ Uçak Bileti Fiyat Öneri Sistemi - Streamlit Arayüzü
========================================================
Kullanıcı rota ve tarih seçer, model o gün için beklenen fiyatı tahmin eder.
Kullanıcının gördüğü fiyatla karşılaştırıp "AL / BEKLE" önerisi verir.

Yeni: Hava durumu bilgisi de gösterilir ve model feature'larına eklenmiştir.
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import date, timedelta
from xgboost import XGBRegressor

# ─── Yollar ───────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "xgb_price_model.json"
FEATURES_CSV = PROJECT_ROOT / "data" / "processed" / "flights_features.csv"
WEATHER_CSV = PROJECT_ROOT / "data" / "processed" / "weather_data.csv"

# ─── Sabitler ─────────────────────────────────────────────────────────────────
BAYRAM_START = date(2026, 5, 27)
BAYRAM_END = date(2026, 5, 30)
PRE_BAYRAM_DAYS = 7
POST_BAYRAM_DAYS = 7

ROUTES = {
    "İstanbul → Antalya": "IST_AYT",
    "Antalya → İstanbul": "AYT_IST",
    "İstanbul → İzmir": "IST_ADB",
    "İzmir → İstanbul": "ADB_IST",
    "İstanbul (SAW) → Ankara": "SAW_ESB",
    "Ankara → İstanbul (SAW)": "ESB_SAW",
}

# Rota -> varış şehri kodu eşlemesi (hava durumu için)
ROUTE_DEST_CITY = {
    "IST_AYT": "AYT",
    "AYT_IST": "IST",
    "IST_ADB": "ADB",
    "ADB_IST": "IST",
    "SAW_ESB": "ESB",
    "ESB_SAW": "IST",
}

# Şehir isimleri
CITY_NAMES = {
    "IST": "İstanbul",
    "AYT": "Antalya",
    "ESB": "Ankara",
    "ADB": "İzmir",
}

# Hava durumu emojileri
WEATHER_EMOJIS = {
    "Acik": "☀️",
    "Az Bulutlu": "🌤️",
    "Parcali Bulutlu": "⛅",
    "Sisli": "🌫️",
    "Ciseleme": "🌦️",
    "Yagmurlu": "🌧️",
    "Karli": "❄️",
    "Firtinali": "🌩️",
    "Bilinmiyor": "❓",
}

# Eğitim verisinden hesaplanan rota istatistikleri
ROUTE_STATS = {
    "IST_AYT": {"global_avg": 4037.58, "global_std": 1346.85, "avg_flights": 42, "avg_airlines": 3},
    "AYT_IST": {"global_avg": 4217.49, "global_std": 963.95, "avg_flights": 37, "avg_airlines": 3},
    "IST_ADB": {"global_avg": 3502.53, "global_std": 919.41, "avg_flights": 34, "avg_airlines": 3},
    "ADB_IST": {"global_avg": 3662.70, "global_std": 959.72, "avg_flights": 36, "avg_airlines": 3},
    "SAW_ESB": {"global_avg": 2331.90, "global_std": 1085.09, "avg_flights": 17, "avg_airlines": 2},
    "ESB_SAW": {"global_avg": 2112.04, "global_std": 584.60, "avg_flights": 12, "avg_airlines": 2},
}


# ─── Model Yükleme ───────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model = XGBRegressor()
    model.load_model(str(MODEL_PATH))
    return model


@st.cache_data
def load_historical_data():
    if FEATURES_CSV.exists():
        df = pd.read_csv(FEATURES_CSV)
        df["depart_date"] = pd.to_datetime(df["depart_date"])
        df["route"] = df["origin"] + "_" + df["destination"]
        return df
    return pd.DataFrame()


@st.cache_data
def load_weather_data():
    """Hava durumu verisini yükle."""
    if WEATHER_CSV.exists():
        df = pd.read_csv(WEATHER_CSV)
        df["date"] = pd.to_datetime(df["date"])
        return df
    return pd.DataFrame()


# ─── Yardımcı Fonksiyonlar ───────────────────────────────────────────────────
def classify_bayram_period(d: date) -> str:
    if BAYRAM_START <= d <= BAYRAM_END:
        return "Bayram"
    elif (BAYRAM_START - timedelta(days=PRE_BAYRAM_DAYS)) <= d < BAYRAM_START:
        return "Bayram Oncesi"
    elif BAYRAM_END < d <= (BAYRAM_END + timedelta(days=POST_BAYRAM_DAYS)):
        return "Bayram Sonrasi"
    else:
        return "Normal"


def get_weather_for_date(weather_df: pd.DataFrame, city_code: str, target_date: date):
    """Belirli bir şehir ve tarih için hava durumu bilgisini getir."""
    if weather_df.empty:
        return None
    
    match = weather_df[
        (weather_df["city_code"] == city_code) & 
        (weather_df["date"].dt.date == target_date)
    ]
    
    if match.empty:
        return None
    
    row = match.iloc[0]
    return {
        "temp_max": row["temp_max"],
        "temp_min": row["temp_min"],
        "precipitation": row["precipitation"],
        "wind_speed": row["wind_speed"],
        "condition": row["weather_condition"],
        "is_bad": row["is_bad_weather"],
    }


def build_features(route_code: str, depart_date: date, weather_df: pd.DataFrame) -> pd.DataFrame:
    """Kullanıcı girdisinden model feature vektörü oluştur."""
    
    # Zaman feature'ları
    today = date(2026, 4, 30)  # Scrape tarihi (veri toplama günü)
    days_to_flight = (depart_date - today).days
    depart_weekday = depart_date.weekday()
    is_weekend = 1 if depart_weekday in [5, 6] else 0
    day_of_month = depart_date.day
    week_of_month = (day_of_month - 1) // 7 + 1

    # Bayram feature'ları
    is_bayram = 1 if BAYRAM_START <= depart_date <= BAYRAM_END else 0
    days_to_bayram = (
        (BAYRAM_START - depart_date).days if depart_date < BAYRAM_START
        else (depart_date - BAYRAM_END).days if depart_date > BAYRAM_END
        else 0
    )
    bayram_proximity = np.exp(-days_to_bayram / 7)
    bayram_period = classify_bayram_period(depart_date)

    # Rota istatistikleri
    stats = ROUTE_STATS[route_code]
    
    # Hava durumu
    dest_city = ROUTE_DEST_CITY[route_code]
    weather_info = get_weather_for_date(weather_df, dest_city, depart_date)
    
    if weather_info:
        temp_max = weather_info["temp_max"]
        temp_min = weather_info["temp_min"]
        precipitation = weather_info["precipitation"]
        wind_speed = weather_info["wind_speed"]
        weather_condition = weather_info["condition"]
        is_bad_weather = weather_info["is_bad"]
    else:
        # Hava durumu yoksa en yaygın değerler
        temp_max = 20.0
        temp_min = 10.0
        precipitation = 0.0
        wind_speed = 10.0
        weather_condition = "Acik"
        is_bad_weather = 0

    # Feature vektörü - model feature_names sırasına göre (34 feature)
    features = [
        ("days_to_flight", days_to_flight),
        ("depart_weekday", depart_weekday),
        ("is_weekend", is_weekend),
        ("day_of_month", day_of_month),
        ("week_of_month", week_of_month),
        ("is_bayram", is_bayram),
        ("days_to_bayram", days_to_bayram),
        ("bayram_proximity", bayram_proximity),
        ("flight_count", stats["avg_flights"]),
        ("airline_count", stats["avg_airlines"]),
        ("route_global_avg", stats["global_avg"]),
        ("route_global_std", stats["global_std"]),
        # Hava durumu (model sırasında route'lardan önce)
        ("temp_max", temp_max),
        ("temp_min", temp_min),
        ("precipitation", precipitation),
        ("wind_speed", wind_speed),
        ("is_bad_weather", is_bad_weather),
        # Route one-hot
        ("route_ADB_IST", 1 if route_code == "ADB_IST" else 0),
        ("route_AYT_IST", 1 if route_code == "AYT_IST" else 0),
        ("route_ESB_SAW", 1 if route_code == "ESB_SAW" else 0),
        ("route_IST_ADB", 1 if route_code == "IST_ADB" else 0),
        ("route_IST_AYT", 1 if route_code == "IST_AYT" else 0),
        ("route_SAW_ESB", 1 if route_code == "SAW_ESB" else 0),
        # Bayram period one-hot (alfabetik sıra: Bayram, Bayram Oncesi, Bayram Sonrasi, Normal)
        ("bayram_period_Bayram", 1 if bayram_period == "Bayram" else 0),
        ("bayram_period_Bayram Oncesi", 1 if bayram_period == "Bayram Oncesi" else 0),
        ("bayram_period_Bayram Sonrasi", 1 if bayram_period == "Bayram Sonrasi" else 0),
        ("bayram_period_Normal", 1 if bayram_period == "Normal" else 0),
        # Weather condition one-hot
        ("weather_condition_Acik", 1 if weather_condition == "Acik" else 0),
        ("weather_condition_Ciseleme", 1 if weather_condition == "Ciseleme" else 0),
        ("weather_condition_Firtinali", 1 if weather_condition == "Firtinali" else 0),
        ("weather_condition_Karli", 1 if weather_condition == "Karli" else 0),
        ("weather_condition_Sisli", 1 if weather_condition == "Sisli" else 0),
        ("weather_condition_Yagmurlu", 1 if weather_condition == "Yagmurlu" else 0),
    ]

    return pd.DataFrame([dict(features)])


def get_recommendation(user_price: float, predicted_price: float, bayram_period: str):
    """Fiyat karşılaştırması yapıp öneri ver."""
    diff_pct = (user_price - predicted_price) / predicted_price * 100

    if bayram_period == "Bayram":
        if diff_pct <= 5:
            return "✅ AL", "success", "Bayram dönemi için makul bir fiyat. Daha fazla düşmesi beklenmez."
        elif diff_pct <= 20:
            return "⏸️ NORMAL", "warning", "Bayram ortalamasında. Acil değilse birkaç gün bekleyebilirsin."
        else:
            return "❌ PAHALI", "error", "Bayram ortalamasının bile üstünde. Kesinlikle bekle."
    else:
        if diff_pct <= -10:
            return "🔥 ÇOK UCUZ - HEMEN AL", "success", "Bu rota için ortalamanın çok altında. Kaçırma!"
        elif diff_pct <= 5:
            return "✅ İYİ FİYAT - AL", "success", "Ortalamanın altında veya civarında. Almak mantıklı."
        elif diff_pct <= 20:
            return "⏸️ NORMAL - BEKLEYEBİLİRSİN", "warning", "Ortalama fiyat. Acil değilse bekle, düşebilir."
        else:
            return "❌ PAHALI - BEKLE", "error", "Ortalamanın çok üstünde. Fiyatlar düşene kadar bekle."


# ─── Sayfa Ayarları ───────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Uçak Bileti Öneri Sistemi",
    page_icon="✈️",
    layout="wide"
)

st.title("✈️ Uçak Bileti Fiyat Öneri Sistemi")
st.markdown("Rota ve tarih seç, gördüğün fiyatı gir → **AL mı BEKLE mi** öğren.")
st.divider()

# ─── Model Yükle ─────────────────────────────────────────────────────────────
if not MODEL_PATH.exists():
    st.error("❌ Model dosyası bulunamadı. Önce modeli eğitin: `python src/models/xgboost_model.py`")
    st.stop()

model = load_model()
hist_df = load_historical_data()
weather_df = load_weather_data()

# ─── Kullanıcı Girdileri ─────────────────────────────────────────────────────
col1, col2, col3 = st.columns([2, 2, 2])

with col1:
    selected_route = st.selectbox("📍 Rota Seçin:", list(ROUTES.keys()))
    route_code = ROUTES[selected_route]

with col2:
    selected_date = st.date_input(
        "📅 Uçuş Tarihi:",
        value=date(2026, 5, 15),
        min_value=date(2026, 5, 1),
        max_value=date(2026, 6, 14),
    )

with col3:
    user_price = st.number_input(
        "💰 Gördüğün Fiyat (TL):",
        min_value=500,
        max_value=20000,
        value=3500,
        step=100,
    )

# ─── Hava Durumu Önizleme ────────────────────────────────────────────────────
if not weather_df.empty:
    dest_city = ROUTE_DEST_CITY[route_code]
    weather_info = get_weather_for_date(weather_df, dest_city, selected_date)
    
    if weather_info:
        emoji = WEATHER_EMOJIS.get(weather_info["condition"], "❓")
        city_name = CITY_NAMES.get(dest_city, dest_city)
        
        st.markdown(
            f"🌤️ **{city_name} Hava Durumu:** "
            f"{emoji} {weather_info['condition']}, "
            f"{weather_info['temp_max']:.0f}°C / {weather_info['temp_min']:.0f}°C, "
            f"💧 {weather_info['precipitation']:.1f}mm, "
            f"💨 {weather_info['wind_speed']:.0f} km/h"
        )

# ─── Tahmin ve Öneri ─────────────────────────────────────────────────────────
if st.button("🔍 Analiz Et", type="primary", use_container_width=True):
    st.divider()

    # Feature oluştur ve tahmin yap
    features_df = build_features(route_code, selected_date, weather_df)
    predicted_price = model.predict(features_df)[0]
    bayram_period = classify_bayram_period(selected_date)

    # Öneri al
    recommendation, status_type, explanation = get_recommendation(
        user_price, predicted_price, bayram_period
    )

    # ─── Sonuç Gösterimi ─────────────────────────────────────────────────────
    st.subheader("📊 Analiz Sonucu")

    # Ana öneri
    if status_type == "success":
        st.success(f"### {recommendation}")
    elif status_type == "warning":
        st.warning(f"### {recommendation}")
    else:
        st.error(f"### {recommendation}")

    st.info(f"💡 {explanation}")

    # Detay metrikleri
    st.divider()
    m1, m2, m3, m4 = st.columns(4)

    diff = user_price - predicted_price
    diff_pct = (diff / predicted_price) * 100

    m1.metric("Model Tahmini", f"{predicted_price:,.0f} TL", help="Bu rota+gün için beklenen medyan fiyat")
    m2.metric("Senin Fiyatın", f"{user_price:,.0f} TL")
    m3.metric("Fark", f"{diff:+,.0f} TL", delta=f"%{diff_pct:+.1f}", delta_color="inverse")
    m4.metric("Dönem", bayram_period, help="Bayram dönemine göre sınıflandırma")

    # ─── Ek Bilgiler ─────────────────────────────────────────────────────────
    st.divider()

    info_col1, info_col2, info_col3 = st.columns(3)

    with info_col1:
        st.subheader("📈 Bu Rota Hakkında")
        stats = ROUTE_STATS[route_code]
        st.write(f"**Rota:** {selected_route}")
        st.write(f"**Genel Ortalama:** {stats['global_avg']:,.0f} TL")
        st.write(f"**Standart Sapma:** {stats['global_std']:,.0f} TL")
        st.write(f"**Günlük Ortalama Uçuş:** {stats['avg_flights']}")
        st.write(f"**Havayolu Sayısı:** {stats['avg_airlines']}")

    with info_col2:
        st.subheader("📅 Tarih Bilgisi")
        weekday_names = ["Pazartesi", "Salı", "Çarşamba", "Perşembe", "Cuma", "Cumartesi", "Pazar"]
        st.write(f"**Tarih:** {selected_date.strftime('%d.%m.%Y')} ({weekday_names[selected_date.weekday()]})")
        st.write(f"**Dönem:** {bayram_period}")
        if bayram_period == "Bayram":
            st.warning("⚠️ Kurban Bayramı dönemi! Fiyatlar normalden ~%47 daha yüksek.")
        elif bayram_period == "Bayram Oncesi":
            st.info("📌 Bayram öncesi dönem. Fiyatlar yükselmeye başlamış olabilir.")
        elif bayram_period == "Bayram Sonrasi":
            st.info("📌 Bayram sonrası dönem. Fiyatlar normale dönüyor.")

    with info_col3:
        st.subheader("🌤️ Hava Durumu")
        if not weather_df.empty:
            dest_city = ROUTE_DEST_CITY[route_code]
            weather_info = get_weather_for_date(weather_df, dest_city, selected_date)
            
            if weather_info:
                emoji = WEATHER_EMOJIS.get(weather_info["condition"], "❓")
                city_name = CITY_NAMES.get(dest_city, dest_city)
                
                st.write(f"**Şehir:** {city_name}")
                st.write(f"**Durum:** {emoji} {weather_info['condition']}")
                st.write(f"**Sıcaklık:** {weather_info['temp_max']:.0f}°C / {weather_info['temp_min']:.0f}°C")
                st.write(f"**Yağış:** {weather_info['precipitation']:.1f} mm")
                st.write(f"**Rüzgar:** {weather_info['wind_speed']:.0f} km/h")
                
                if weather_info["is_bad"]:
                    st.warning("⚠️ Kötü hava koşulu! Fiyatlar etkilenebilir.")
                elif weather_info["condition"] == "Acik":
                    st.success("✅ Güzel hava! Tatil rotalarında talep artabilir.")
            else:
                st.write("Bu tarih için hava durumu verisi bulunamadı.")
        else:
            st.write("Hava durumu verisi yüklenemedi.")

    # ─── Geçmiş Veri Grafiği ─────────────────────────────────────────────────
    if not hist_df.empty:
        st.divider()
        st.subheader("📉 Bu Rotanın Fiyat Geçmişi")

        route_hist = hist_df[hist_df["route"] == route_code].copy()
        if not route_hist.empty:
            import matplotlib.pyplot as plt

            daily = route_hist.groupby("depart_date")["price"].agg(["median", "min", "max"]).reset_index()

            fig, ax = plt.subplots(figsize=(12, 4))
            ax.plot(daily["depart_date"], daily["median"], color="#2E86C1", linewidth=2, label="Medyan Fiyat")
            ax.fill_between(daily["depart_date"], daily["min"], daily["max"], alpha=0.15, color="#2E86C1")

            # Seçilen tarihi işaretle
            ax.axvline(pd.Timestamp(selected_date), color="red", linestyle="--", linewidth=1.5, label="Seçilen Tarih")

            # Bayram bölgesi
            ax.axvspan(pd.Timestamp(BAYRAM_START), pd.Timestamp(BAYRAM_END),
                       alpha=0.15, color="orange", label="Bayram Dönemi")

            # Tahmin noktası
            ax.scatter([pd.Timestamp(selected_date)], [predicted_price],
                       color="red", s=100, zorder=5, label=f"Tahmin: {predicted_price:,.0f} TL")

            ax.set_xlabel("Kalkış Tarihi")
            ax.set_ylabel("Fiyat (TL)")
            ax.set_title(f"{selected_route} - Fiyat Trendi")
            ax.legend(loc="upper left")
            ax.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

# ─── Sidebar ─────────────────────────────────────────────────────────────────
st.sidebar.header("ℹ️ Sistem Bilgisi")
st.sidebar.markdown("""
**Model:** XGBoost Regresyon  
**Eğitim Verisi:** 6 rota, 45 gün (~9K uçuş)  
**Performans:** R² = 0.75, MAE = ±442 TL  
**Veri Kaynağı:** Enuygun.com  
**Hava Durumu:** Open-Meteo API  
""")

st.sidebar.divider()
st.sidebar.markdown("""
**Nasıl Çalışır?**
1. Rota ve tarih seçin
2. Gördüğünüz bilet fiyatını girin
3. Model o gün için beklenen fiyatı tahmin eder
4. Hava durumu da dikkate alınır
5. Fiyatınız tahmine göre ucuz/pahalı mı değerlendirilir
""")

st.sidebar.divider()
st.sidebar.caption("Veri Madenciliği Dönem Projesi | 2026")
