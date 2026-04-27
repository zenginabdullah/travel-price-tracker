import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import subprocess
import sys

#sayfa ayarları
st.set_page_config(
    page_title="Uçak Bileti Öneri Sistemi", 
    page_icon="✈️", 
    layout="wide"
)

#dosya yolları
DATA_PATH = Path("data/processed/flights_features.csv")

#veri yükleme
@st.cache_data
def load_data():
    if DATA_PATH.exists():
        df = pd.read_csv(DATA_PATH)
        # rota sütu
        if "origin" in df.columns and "destination" in df.columns:
            df["rota"] = df["origin"] + " ➔ " + df["destination"]
        return df
    return pd.DataFrame()

#başlık
st.title("Akıllı Uçak Bileti Fiyat Takip ve Öneri Sistemi")
st.markdown("Veri Madenciliği Dönem Projesi")

#sistem yönetimi
st.sidebar.header("Sistem Yönetimi")
if st.sidebar.button("Verileri Şimdi Güncelle"):
    with st.spinner("Çalışıyor... Lütfen bekleyin."):
        try:
            # 1. Adım: Veri Kazıma
            st.sidebar.info("1/3: Veri kazınıyor...")
            subprocess.run([sys.executable, "src/collectors/flight_scraper_enuygun.py"], check=True)
            
            # 2. Adım: Veri Temizleme
            st.sidebar.info("2/3: Veri temizleniyor...")
            subprocess.run([sys.executable, "src/clean/clean_flights_enuygun.py"], check=True)
            
            # 3. Adım: Özellik Mühendisliği
            st.sidebar.info("3/3: Özellikler türetiliyor...")
            subprocess.run([sys.executable, "src/analysis/feature_engineering.py"], check=True)
            
            # Önbelleği temizle ve sayfayı yenile
            st.cache_data.clear()
            st.sidebar.success("Sistem başarıyla güncellendi!")
            st.rerun()
            
        except Exception as e:
            st.sidebar.error(f"Hata oluştu: {e}")

st.sidebar.divider()

#veriyi yükle
df = load_data()

if df.empty:
    st.warning("Henüz işlenmiş veri bulunamadı. Lütfen soldaki 'Verileri Şimdi Güncelle' butonuna basarak ilk veri çekme işlemini başlatın.")
else:
    #filtreler
    st.sidebar.header("Uçuş Filtreleri")
    
    # 1. Rota Seçimi
    rotalar = sorted(df["rota"].unique())
    secilen_rota = st.sidebar.selectbox("Uçuş Rotası:", rotalar)
    
    # Veriyi rotaya göre daralt
    rota_df = df[df["rota"] == secilen_rota]
    
    # 2. Havayolu Seçimi
    havayollari = sorted(rota_df["airline"].unique())
    secilen_havayollari = st.sidebar.multiselect(
        "Havayolu Şirketi:", 
        havayollari, 
        default=havayollari
    )
    
    # Ana filtreleme
    filtered_df = rota_df[rota_df["airline"].isin(secilen_havayollari)]

    #metrikler
    if filtered_df.empty:
        st.error("Seçilen filtrelere uygun veri bulunamadı.")
    else:
        m1, m2, m3 = st.columns(3)
        m1.metric("Toplam Kayıt", f"{len(filtered_df)} Uçuş")
        m2.metric("En Ucuz Bilet", f"{filtered_df['price'].min():,.0f} TL")
        m3.metric("Ortalama Fiyat", f"{filtered_df['price'].mean():,.0f} TL")

        st.divider()

        #grafikler
        g1, g2 = st.columns(2)

        with g1:
            st.subheader("Uçuşa Kalan Güne Göre Fiyat Trendi")
            #günlük ortalama fiyatlar
            trend_df = filtered_df.groupby("days_to_flight")["price"].mean().reset_index()
            
            fig1, ax1 = plt.subplots(figsize=(10, 5))
            ax1.plot(trend_df["days_to_flight"], trend_df["price"], marker='o', linestyle='-', color='#2E86C1')
            ax1.set_xlabel("Uçuşa Kalan Gün Sayısı")
            ax1.set_ylabel("Ortalama Fiyat (TL)")
            ax1.invert_xaxis()
            ax1.grid(True, alpha=0.2)
            st.pyplot(fig1)
            st.caption("Grafiğin sağ tarafı uçuş tarihine en yakın zamanı gösterir.")

        with g2:
            st.subheader("Günün Saat Dilimine Göre Fiyatlar")
            #saat dilimi bazlı ortalama
            zaman_df = filtered_df.groupby("time_of_day")["price"].mean().reindex(["Sabah", "Öğle", "Akşam", "Gece"]).reset_index()
            
            fig2, ax2 = plt.subplots(figsize=(10, 5))
            ax2.bar(zaman_df["time_of_day"], zaman_df["price"], color='#AED6F1')
            ax2.set_xlabel("Saat Dilimi")
            ax2.set_ylabel("Ortalama Fiyat (TL)")
            st.pyplot(fig2)
            st.caption("Uçuş saatlerine göre bilet fiyatlarının genel dağılımı.")

        st.divider()

        #veri tablosu
        with st.expander("İşlenmiş Verileri İncele"):
            st.dataframe(filtered_df.sort_values("scrape_ts", ascending=False), width='stretch')

#alt bilgi
st.sidebar.markdown("---")
st.sidebar.info("**İpucu:** Veriler güncellenirken tarayıcıyı kapatmayın.")