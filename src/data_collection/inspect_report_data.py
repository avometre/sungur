"""Rapor 10 için örnekleri ve istatistikleri toplayıp ekrana basan modül.
"""

import csv
import json
from pathlib import Path


def inspect_all():
    print("=== 1. WIKIDATA (İlk 10) ===")
    p_wiki = Path("data/interim/wikidata_ota_tr.jsonl")
    if p_wiki.exists():
        with open(p_wiki, encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= 10:
                    break
                d = json.loads(line)
                print(f"{i+1}. QID: {d['qid']} -> Arap: {d['arap']} -> Latin: {d['latin']} (Tür: {d['tur']})")

    print("\n=== 2. TR.WIKTIONARY (İlk 10) ===")
    p_wikt = Path("data/interim/tr_wiktionary_ota.jsonl")
    if p_wikt.exists():
        with open(p_wikt, encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= 10:
                    break
                d = json.loads(line)
                print(f"{i+1}. Arap: {d['arap']} -> Latin: {d['latin']} -> Anlam: {d['anlam_tr'][:60]}")

    print("\n=== 3. TR.WIKIPEDIA (İlk 10) ===")
    p_wp = Path("data/interim/tr_wikipedia_ota.jsonl")
    if p_wp.exists():
        with open(p_wp, encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= 10:
                    break
                d = json.loads(line)
                print(f"{i+1}. Sayfa: {d['sayfa']} -> Arap: {d['arap']} -> Latin: {d['latin']}")

    print("\n=== 4. SADE SOZLUK (İlk 10) ===")
    p_sade = Path("data/interim/sade_sozluk.jsonl")
    if p_sade.exists():
        with open(p_sade, encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= 10:
                    break
                d = json.loads(line)
                print(f"{i+1}. Kelime: {d['kelime_latin']} -> Anlam: {d['anlam_tr'][:60]}")

    print("\n=== 5. INCIL PARALELI (10 ornek) ===")
    p_bible = Path("data/interim/ota_bible_sample.jsonl")
    if p_bible.exists():
        with open(p_bible, encoding="utf-8") as f:
            for i, line in enumerate(f):
                d = json.loads(line)
                print(f"{i+1}. Ref: {d['ref']}")
                print(f"   Osmanlıca (Arap): {d['osmanlica_arap']}")
                print(f"   Modern TR (İncil): {d['turkce_modern']}")
                print(f"   İngilizce: {d['ingilizce']}")

    print("\n=== 6. ATATÜRK SADELEŞTİRME (10 ornek) ===")
    p_ataturk = Path("data/interim/vikikaynak_sadelestirilmis.jsonl")
    if p_ataturk.exists():
        count = 0
        with open(p_ataturk, encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                if d.get("latin") != d.get("latin_sade"):
                    count += 1
                    print(f"{count}. Başlık: {d['baslik']}")
                    print(f"   Orijinal: {d['latin']}")
                    print(f"   Sade:     {d['latin_sade']}")
                    if count >= 10:
                        break

    print("\n=== 7. TOP 15 LATOC ASIL ESLESMESI ===")
    p_latoc = Path("data/interim/latoc_asil_eslesme.csv")
    if p_latoc.exists():
        with open(p_latoc, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = sorted(list(reader), key=lambda x: float(x["skor"]), reverse=True)
            for i, r in enumerate(rows[:15]):
                print(f"{i+1}. Skor: {r['skor']} ({r['eslesme_guveni']}) | LATOC: {r['latoc_kitap']} | Kaynak: {r['kaynak']} | ID: {r['kimlik']}")


if __name__ == "__main__":
    inspect_all()
