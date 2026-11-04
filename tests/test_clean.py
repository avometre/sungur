"""clean_raw temizleme mantığı için birim testleri."""

from src.clean.clean_raw import (
    clean_gazetteer,
    clean_wikisource,
    clean_wiktionary,
    extract_gazetteer_candidates,
)


class TestCleanWikisource:
    def test_drops_kurdish_pages(self) -> None:
        records = [
            # Kürtçe başlıklı sayfa (U+06CE ێ)
            {"id": "ws:100:0", "baslik": "دیوانی \u06ce", "arap_ham": "مصرع بیر", "arap": "مصرع بیر"},
            {"id": "ws:100:1", "baslik": "دیوانی \u06ce", "arap_ham": "مصرع ایکی", "arap": "مصرع ایکی"},
            # Normal Osmanlıca sayfa
            {"id": "ws:200:0", "baslik": "کتاب", "arap_ham": "اول کلام", "arap": "اول کلام"},
            {"id": "ws:200:1", "baslik": "کتاب", "arap_ham": "آخیر کلام", "arap": "آخیر کلام"},
        ]
        kept, excluded, stats = clean_wikisource(records)
        assert len(kept) == 2
        assert len(excluded) == 2
        assert all(ex["neden"] == "kurtce_sayfa" for ex in excluded)
        assert kept[0]["id"] == "ws:200:0"
        assert kept[0]["sira"] == 0
        assert kept[1]["sira"] == 1

    def test_drops_short_and_latin_heavy_lines(self) -> None:
        records = [
            # Çok kısa (3 Arapça harften az: sayfa no "۳")
            {"id": "ws:200:0", "baslik": "کتاب", "arap_ham": "۳", "arap": "۳"},
            # Latin ağırlıklı satır
            {"id": "ws:200:1", "baslik": "کتاب", "arap_ham": "sayfa başlığı کتاب", "arap": "sayfa başlığı کتاب"},
            # Geçerli satır
            {"id": "ws:200:2", "baslik": "کتاب", "arap_ham": "دولت علیه عثمانیه", "arap": "دولت علیه عثمانیه"},
        ]
        kept, excluded, stats = clean_wikisource(records)
        assert len(kept) == 1
        assert kept[0]["id"] == "ws:200:2"
        reasons = [ex["neden"] for ex in excluded]
        assert "cok_kisa" in reasons
        assert "latin_agirlikli" in reasons

    def test_drops_intra_page_duplicates(self) -> None:
        records = [
            {"id": "ws:200:0", "baslik": "کتاب", "arap_ham": "دولت علیه عثمانیه", "arap": "دولت علیه عثمانیه"},
            {"id": "ws:200:1", "baslik": "کتاب", "arap_ham": "دولت علیه عثمانیه", "arap": "دولت علیه عثمانیه"},
        ]
        kept, excluded, stats = clean_wikisource(records)
        assert len(kept) == 1
        assert len(excluded) == 1
        assert excluded[0]["neden"] == "tekrar"


class TestCleanWiktionary:
    def test_drops_missing_latin(self) -> None:
        records = [
            {"id": "wikt:1", "arap_ham": "کتاب", "arap": "کتاب", "latin": "", "tur": "noun"},
            {"id": "wikt:2", "arap_ham": "کتاب", "arap": "کتاب", "latin": "kitap", "latin_sade": "kitap", "tur": "noun", "anlam": ["a"]},
        ]
        kept, excluded, stats = clean_wiktionary(records)
        assert len(kept) == 1
        assert len(excluded) == 1
        assert excluded[0]["neden"] == "latin_yok"

    def test_merges_duplicates_and_marks_suspicious(self) -> None:
        records = [
            {"id": "wikt:1", "arap_ham": "کتاب", "arap": "کتاب", "latin": "kitap", "latin_sade": "kitap", "tur": "noun", "anlam": ["kitap1"]},
            {"id": "wikt:2", "arap_ham": "کتاب", "arap": "کتاب", "latin": "kitap", "latin_sade": "kitap", "tur": "noun", "anlam": ["kitap2"]},
            # Düşük benzerlikli (şüpheli) kayıt
            {"id": "wikt:3", "arap_ham": "آیا", "arap": "آیا", "latin": "aya", "latin_sade": "aya", "tur": "noun", "anlam": ["avuç içi"]},
        ]
        kept, excluded, stats = clean_wiktionary(records)
        assert len(kept) == 2
        kitap_rec = [r for r in kept if r["latin"] == "kitap"][0]
        assert "kitap1" in kitap_rec["anlam"] and "kitap2" in kitap_rec["anlam"]
        assert kitap_rec["supheli"] is False

        aya_rec = [r for r in kept if r["latin"] == "aya"][0]
        assert aya_rec["supheli"] is True


class TestCleanGazetteer:
    def test_extract_candidates(self) -> None:
        cands, belirsiz = extract_gazetteer_candidates("Abadan/Hazar (el-Hazra)")
        assert cands == ["Abadan", "Hazar", "el-Hazra"]
        assert belirsiz is False

        cands2, belirsiz2 = extract_gazetteer_candidates("İpsili (?) / Kanlıada")
        assert cands2 == ["İpsili", "Kanlıada"]
        assert belirsiz2 is True

    def test_selects_best_candidate_by_skeleton(self) -> None:
        records = [
            # "\u0642\u0627\u0646\u0644\u06cc \u0627\u0637\u0647" (قانلی اطه): Kanlıada seçilmeli, İpsili elenmeli
            {
                "id": "gaz:1",
                "arap_ham": "\u0642\u0627\u0646\u0644\u06cc \u0627\u0637\u0647",
                "arap": "\u0642\u0627\u0646\u0644\u06cc \u0627\u0637\u0647",
                "latin": "İpsili / Kanlıada",
                "tur": "ada",
            }
        ]
        kept, excluded, stats = clean_gazetteer(records)
        assert len(kept) == 1
        rec = kept[0]
        assert rec["latin"] == "Kanlıada"
        assert rec["eslesme_skoru"] >= 0.8
        assert rec["guvenilir"] is True
        assert rec["latin_adaylar"] == ["İpsili", "Kanlıada"]
