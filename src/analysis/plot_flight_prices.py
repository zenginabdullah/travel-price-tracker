from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

INPUT_CSV = Path("data/processed/flights_enuygun_clean.csv")
OUT_DIR = Path("reports")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def main():
    df = pd.read_csv(INPUT_CSV)
    df["scrape_ts"] = pd.to_datetime(df["scrape_ts"], errors="coerce", utc=True)
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df = df.dropna(subset=["scrape_ts", "price"]).copy()

    df["day"] = df["scrape_ts"].dt.tz_convert("Europe/Istanbul").dt.date

    daily = (
        df.groupby("day", as_index=False)
          .agg(
              min_price=("price", "min"),
              median_price=("price", "median"),
              max_price=("price", "max"),
              sample_size=("price", "count"),
          )
          .sort_values("day")
    )

    daily["day"] = pd.to_datetime(daily["day"])  #matplotlib için

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(daily["day"], daily["min_price"], marker="o", label="Günlük Minimum")
    ax.plot(daily["day"], daily["median_price"], marker="o", label="Günlük Medyan")
    ax.plot(daily["day"], daily["max_price"], marker="o", label="Günlük Maksimum")

    ax.set_title("IST-AYT Uçuş Fiyat Takibi (Enuygun)")
    ax.set_xlabel("Tarih")
    ax.set_ylabel("Fiyat (TRY)")
    ax.grid(alpha=0.3)
    ax.legend()

    #Tarih formatını sıkılaştır
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    plt.xticks(rotation=30)
    plt.tight_layout()

    out_png = OUT_DIR / "flight_prices_daily_summary.png"
    out_csv = OUT_DIR / "flight_prices_daily_summary.csv"
    plt.savefig(out_png, dpi=140)
    daily.to_csv(out_csv, index=False, encoding="utf-8")
    plt.close()

    print(f"[OK] Grafik: {out_png}")
    print(f"[OK] Özet: {out_csv}")

if __name__ == "__main__":
    main()