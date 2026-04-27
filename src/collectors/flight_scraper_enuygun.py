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

SOURCE = "enuygun"
CURRENCY = "TRY"

ROUTES = [
    {
        "origin": "IST", 
        "destination": "AYT", 
        "depart_date": "2026-05-07",
        "return_date": "2026-05-09",
        "url": "https://www.enuygun.com/ucak-bileti/arama/istanbul-antalya-havalimani-ista-ayt/?gidis=07.05.2026&donus=09.05.2026&yetiskin=1&sinif=ekonomi&currency=TRY&save=1&ref=homepage&geotrip=domestic&trip=domestic"
    },
    {
        "origin": "SAW", 
        "destination": "ESB", 
        "depart_date": "2026-05-07",
        "return_date": "2026-05-09",
        "url": "https://www.enuygun.com/ucak-bileti/arama/istanbul-sabiha-gokcen-havalimani-ankara-esenboga-havalimani-saw-esb/?gidis=07.05.2026&donus=09.05.2026&yetiskin=1&sinif=ekonomi&currency=TRY&save=1"
    },
    {
        "origin": "IST", 
        "destination": "ADB", 
        "depart_date": "2026-05-07",
        "return_date": "2026-05-09",
        "url": "https://www.enuygun.com/ucak-bileti/arama/istanbul-izmir-adnan-menderes-havalimani-ista-adb/?gidis=07.05.2026&donus=09.05.2026&yetiskin=1&sinif=ekonomi&currency=TRY&save=1"
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
    times = re.findall(r"\b([0-2]\d:[0-5]\d)\b", text)
    if len(times) >= 2:
        return times[0], times[1]
    if len(times) == 1:
        return times[0], ""
    return "", ""

def scrape_enuygun(url: str, origin: str, destination: str, depart_date: str, return_date: str) -> list[dict]:
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

        cards = page.locator('[data-testid*="flight"]')
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

# --- TERMİNATÖR HAVAYOLU BULUCU (HTML Brute-Force) ---
            airline = "Unknown"
            try:
                # 1. Önce kartın tüm arka plan HTML'ini alıp küçük harfe çevirelim
                html_content = card.inner_html().lower()
                
                # 2. HTML'in içinde havayolu isimleri veya havayolu kodları geçiyor mu bakalım
                # (Sıralama önemli: Önce spesifik olanlar)
                if "türk hava yolları" in html_content or "thy" in html_content or "tk.png" in html_content:
                    airline = "THY"
                elif "pegasus" in html_content or "pc.png" in html_content:
                    airline = "Pegasus"
                elif "ajet" in html_content or "vf.png" in html_content:
                    airline = "AJet"
                elif "sunexpress" in html_content or "xq.png" in html_content:
                    airline = "SunExpress"
                elif "anadolujet" in html_content:
                    airline = "AnadoluJet"
                    
                # 3. Hala bulamadıysa, kartın düz metninde son bir şansımızı deneyelim
                if airline == "Unknown":
                    airline = extract_airline_from_text(txt)
            except Exception as e:
                pass
            # -----------------------------------------------------------

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

def main():
    all_rows = []
    rid = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print("🚀 Çoklu Rota Veri Kazıma İşlemi Başlıyor...")
    
    #Tüm rotaları döngüye alıyoruz
    for route in ROUTES:
        print(f"\n[INFO] Sıradaki Rota: {route['origin']} -> {route['destination']}")
        rows = scrape_enuygun(
            url=route["url"],
            origin=route["origin"],
            destination=route["destination"],
            depart_date=route["depart_date"],
            return_date=route["return_date"]
        )
        all_rows.extend(rows)
        
        #Sunucuyu yormamak ve ban yememek için rotalar arası bekleme
        jitter(3.0, 6.0)

    out = RAW_DIR / f"flights_enuygun_{rid}.json"
    out.write_text(json.dumps(all_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    
    print(f"\n[OK] İşlem Tamamlandı! Toplam çekilen kayıt: {len(all_rows)}")
    print(f"[OK] Tüm rotalar tek dosyaya kaydedildi: {out}")

if __name__ == "__main__":
    main()