from __future__ import annotations
import json
import re
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError

RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)

TARGET_URL = "https://www.enuygun.com/ucak-bileti/arama/istanbul-antalya-havalimani-ista-ayt/?gidis=07.05.2026&donus=09.05.2026&yetiskin=1&sinif=ekonomi&currency=TRY&save=1&ref=homepage&geotrip=domestic&trip=domestic"

SOURCE = "enuygun"
ORIGIN = "IST"
DESTINATION = "AYT"
DEPART_DATE = "2026-05-07"
RETURN_DATE = "2026-05-09"
CURRENCY = "TRY"

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
        # sadece binlik ayıracı olabilir
        t = t.replace(".", "")
    try:
        return float(t)
    except ValueError:
        return None

def extract_price_from_text(text: str) -> Optional[float]:
    """
    Kart metninden fiyat yakalar.
    Örnek eşleşmeler:
      - 1.478 TL
      - 1478 TL
      - ₺1.478
      - 1.478,50 TL
    """
    if not text:
        return None

    # Önce "sayı + TL"
    m = re.search(r"(\d{1,3}(?:\.\d{3})*(?:,\d{1,2})?|\d+(?:,\d{1,2})?)\s*TL\b", text, flags=re.IGNORECASE)
    if m:
        return parse_tr_number_to_float(m.group(1))

    # Sonra "₺ + sayı"
    m = re.search(r"₺\s*(\d{1,3}(?:\.\d{3})*(?:,\d{1,2})?|\d+(?:,\d{1,2})?)", text, flags=re.IGNORECASE)
    if m:
        return parse_tr_number_to_float(m.group(1))

    return None

def extract_airline_from_text(text: str) -> str:
    known = [
        "Pegasus", "Türk Hava Yolları", "THY", "AJet",
        "SunExpress", "AnadoluJet"
    ]
    for k in known:
        if k.lower() in text.lower():
            return k
    return "Unknown"

def extract_times_from_text(text: str) -> tuple[str, str]:
    # örn: 00:10 ... 01:30
    times = re.findall(r"\b([0-2]\d:[0-5]\d)\b", text)
    if len(times) >= 2:
        return times[0], times[1]
    if len(times) == 1:
        return times[0], ""
    return "", ""

def scrape_enuygun() -> list[dict]:
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

        page.goto(TARGET_URL, wait_until="domcontentloaded", timeout=120000)
        jitter(2.0, 3.5)

        # cookie popup vs.
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

        # sonuçlar yüklensin
        found = None
        for rs in ['[data-testid*="flight"]', '[class*="flight"]', 'article', 'li']:
            try:
                page.wait_for_selector(rs, timeout=10000)
                found = rs
                break
            except PlaywrightTimeoutError:
                continue

        if not found:
            print("[WARN] Sonuç elemanı bulunamadı.")
            context.close()
            browser.close()
            return rows

        cards = page.locator('[data-testid*="flight"]')
        count = cards.count()
        print(f"[INFO] Card selector: [data-testid*='flight'] | count={count}")

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

            airline = extract_airline_from_text(txt)
            dep_time, arr_time = extract_times_from_text(txt)

            key = (airline, dep_time, arr_time, price)
            if key in seen:
                continue
            seen.add(key)

            rows.append({
                "scrape_ts": now_ts(),
                "source": SOURCE,
                "origin": ORIGIN,
                "destination": DESTINATION,
                "depart_date": DEPART_DATE,
                "return_date": RETURN_DATE,
                "airline": airline,
                "depart_time": dep_time,
                "arrival_time": arr_time,
                "price": price,
                "currency": CURRENCY,
                "raw_url": TARGET_URL,
                "run_id": rid
            })

        context.close()
        browser.close()

    return rows

def main():
    rows = scrape_enuygun()
    rid = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = RAW_DIR / f"flights_enuygun_{rid}.json"
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] Enuygun kayıt sayısı: {len(rows)}")
    print(f"[OK] Kaydedildi: {out}")

if __name__ == "__main__":
    main()