from __future__ import annotations
import json
import re
import random
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError

RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)

SOURCE = "enuygun"
CURRENCY = "TRY"
START_DATE = date(2026, 5, 1)
END_DATE = date(2026, 5, 31)
DATE_FORMAT = "%d.%m.%Y"

ROUTES = [
    {
        "origin": "IST", 
        "destination": "AYT", 
        "path": "istanbul-antalya-havalimani-ista-ayt"
    },
    {
        "origin": "AYT", 
        "destination": "IST", 
        "path": "antalya-havalimani-istanbul-ayt-ista"
    },
    {
        "origin": "SAW", 
        "destination": "ESB", 
        "path": "istanbul-sabiha-gokcen-havalimani-ankara-esenboga-havalimani-saw-esb"
    },
    {
        "origin": "ESB", 
        "destination": "SAW", 
        "path": "ankara-esenboga-havalimani-istanbul-sabiha-gokcen-havalimani-esb-saw"
    },
    {
        "origin": "IST", 
        "destination": "ADB", 
        "path": "istanbul-izmir-adnan-menderes-havalimani-ista-adb"
    },
    {
        "origin": "ADB", 
        "destination": "IST", 
        "path": "izmir-adnan-menderes-havalimani-istanbul-adb-ista"
    }
]

def now_ts() -> str:
    return datetime.now(timezone.utc).isoformat()

def make_run_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def jitter(a=1.0, b=2.5):
    time.sleep(random.uniform(a, b))

def parse_tr_number_to_float(num_text: str) -> Optional[float]:
    """'1.478' -> 1478.0, '1.478,50' -> 1478.50"""
    if not num_text:
        return None
    t = num_text.strip()
    if "," in t and "." in t:
        t = t.replace(".", "").replace(",", ".")
    elif "," in t:
        t = t.replace(",", ".")
    else:
        t = t.replace(".", "")
    try:
        return float(t)
    except ValueError:
        return None

def extract_price_from_text(text: str) -> Optional[float]:
    if not text:
        return None
    m = re.search(r"(\d{1,3}(?:\.\d{3})*(?:,\d{1,2})?|\d+(?:,\d{1,2})?)\s*TL\b", text, flags=re.IGNORECASE)
    if m:
        return parse_tr_number_to_float(m.group(1))
    m = re.search(r"₺\s*(\d{1,3}(?:\.\d{3})*(?:,\d{1,2})?|\d+(?:,\d{1,2})?)", text, flags=re.IGNORECASE)
    if m:
        return parse_tr_number_to_float(m.group(1))
    return None

def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()

def extract_airline_from_text(text: str) -> str:
    normalized = normalize_text(text)
    airline_aliases = {
        "THY": ["türk hava yolları", "turkish airlines", "thy", "tk"],
        "Pegasus": ["pegasus", "pc"],
        "AJet": ["ajet", "vf"],
        "SunExpress": ["sunexpress", "xq"],
        "AnadoluJet": ["anadolujet", "anadolu jet"],
    }
    for airline, aliases in airline_aliases.items():
        if any(alias in normalized for alias in aliases):
            return airline
    return "Unknown"

def extract_times_from_text(text: str) -> tuple[str, str]:
    times = re.findall(r"\b([0-2]\d:[0-5]\d)\b", text)
    if len(times) >= 2:
        return times[0], times[1]
    if len(times) == 1:
        return times[0], ""
    return "", ""

def format_enuygun_date(value: date) -> str:
    return value.strftime(DATE_FORMAT)

def build_search_url(path: str, depart_date: date, return_date: Optional[date] = None) -> str:
    query = [
        f"gidis={format_enuygun_date(depart_date)}",
        "yetiskin=1",
        "sinif=ekonomi",
        f"currency={CURRENCY}",
        "save=1",
        "ref=homepage",
        "geotrip=domestic",
        "trip=domestic",
    ]
    if return_date is not None:
        query.insert(1, f"donus={format_enuygun_date(return_date)}")
    return f"https://www.enuygun.com/ucak-bileti/arama/{path}/?{'&'.join(query)}"

def scrape_enuygun(url: str, origin: str, destination: str, depart_date: str, return_date: str = "") -> list[dict]:
    rows: list[dict] = []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(
            locale="tr-TR",
            timezone_id="Europe/Istanbul",
            user_agent=("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36")
        )
        page = context.new_page()

        page.goto(url, wait_until="domcontentloaded", timeout=120000)
        jitter(2.0, 3.5)

        for sel in [
            'button:has-text("Kabul et")',
            'button:has-text("Tümünü kabul et")',
            'button:has-text("Anladım")',
            '[id*="accept"]'
        ]:
            try:
                el = page.locator(sel).first
                if el.count() > 0 and el.is_visible():
                    el.click(timeout=1500)
                    jitter(0.3, 0.8)
            except Exception:
                pass

        found = None
        for rs in ['[data-testid*="flight"]', '[class*="flight"]', 'article', 'li']:
            try:
                page.wait_for_selector(rs, timeout=10000)
                found = rs
                break
            except PlaywrightTimeoutError:
                continue

        if not found:
            print(f"[WARN] Sonuç elemanı bulunamadı: {origin}-{destination}")
            context.close()
            browser.close()
            return rows

        cards = page.locator('.flight-item__wrapper')
        count = cards.count()
        print(f"[INFO] {origin}-{destination} için incelenen kart sayısı: {count}")

        rid = make_run_id()
        seen = set()

        for i in range(min(count, 300)):
            card = cards.nth(i)
            try:
                txt = card.inner_text(timeout=1500).strip()
            except Exception:
                continue

            if not txt:
                continue

            price = extract_price_from_text(txt)
            if price is None:
                continue

            airline = "Unknown"
            try:
                airline = extract_airline_from_text(txt)
                if airline == "Unknown":
                    airline = extract_airline_from_text(card.inner_html())
            except Exception:
                pass

            dep_time, arr_time = extract_times_from_text(txt)

            key = (airline, dep_time, arr_time, price)
            if key in seen:
                continue
            seen.add(key)

            rows.append({
                "scrape_ts": now_ts(),
                "source": SOURCE,
                "origin": origin,
                "destination": destination,
                "depart_date": depart_date,
                "return_date": return_date,
                "airline": airline,
                "depart_time": dep_time,
                "arrival_time": arr_time,
                "price": price,
                "currency": CURRENCY,
                "raw_url": url,
                "run_id": rid
            })

        context.close()
        browser.close()

    return rows

def daterange(start_date: date, end_date: date):
    current = start_date
    while current <= end_date:
        yield current
        current += timedelta(days=1)

def main():
    total_rows = 0

    print("🚀 Çoklu rota ve tarih aralığı veri kazıma işlemi başlıyor...")
    print(f"[INFO] Tarih aralığı: {START_DATE} -> {END_DATE}")

    for travel_date in daterange(START_DATE, END_DATE):
        travel_date_str = travel_date.strftime("%Y%m%d")
        day_dir = RAW_DIR / travel_date_str
        day_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n[INFO] İşlenen gün: {travel_date}")

        for route in ROUTES:
            url = build_search_url(route["path"], travel_date)

            print(f"[INFO] Rota: {route['origin']} -> {route['destination']}")
            rows = scrape_enuygun(
                url=url,
                origin=route["origin"],
                destination=route["destination"],
                depart_date=travel_date.isoformat(),
                return_date=""
            )

            out = day_dir / f"{route['origin']}_{route['destination']}.json"
            out.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

            total_rows += len(rows)
            print(f"[OK] Yazıldı: {out} ({len(rows)} kayıt)")

            jitter(3.0, 6.0)

    print(f"\n[OK] İşlem Tamamlandı! Toplam çekilen kayıt: {total_rows}")

if __name__ == "__main__":
    main()