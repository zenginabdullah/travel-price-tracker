from __future__ import annotations
import json
from pathlib import Path
import pandas as pd

RAW_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "raw"
PROCESSED_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

def load_raw_files() -> pd.DataFrame:
    files = sorted(RAW_DIR.rglob("*.json"))
    rows = []
    for f in files:
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            if isinstance(data, list):
                rows.extend(data)
        except Exception:
            continue
    return pd.DataFrame(rows)

def main():
    df = load_raw_files()
    if df.empty:
        print("[WARN] flights_enuygun_*.json bulunamadı veya boş.")
        return

    # tip dönüşümleri
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df = df[df["price"] > 0].copy()

    if "scrape_ts" in df.columns:
        df["scrape_ts"] = pd.to_datetime(df["scrape_ts"], errors="coerce", utc=True)

    # boş airline/time alanlarını koru ama standardize et
    for col in ["airline", "depart_time", "arrival_time"]:
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str).str.strip()

    # return_date kullanılmıyor (tek yön), sütunu düşür
    if "return_date" in df.columns:
        df = df.drop(columns=["return_date"])

    # duplicate azaltma
    dedup_cols = [c for c in ["run_id", "price", "depart_date", "origin", "destination", "airline", "depart_time", "arrival_time"] if c in df.columns]
    if dedup_cols:
        df = df.drop_duplicates(subset=dedup_cols)

    out_csv = PROCESSED_DIR / "flights_enuygun_clean.csv"
    df.to_csv(out_csv, index=False, encoding="utf-8")

    print(f"[OK] Clean kayıt: {len(df)}")
    print(f"[OK] Yazıldı: {out_csv}")

if __name__ == "__main__":
    main()