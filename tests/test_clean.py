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

    def test_drops_records_without_arabic_letters(self) -> None:
        records = [
            # Rakam girdisi (U+06F9 Doğu Arap-Hint 9)
            {"id": "wikt:9", "arap_ham": "\u06f9", "arap": "9", "latin": "dokuz", "tur": "num"},
            # Rakam girdisi (U+0664 Arap-Hint 4)
            {"id": "wikt:4", "arap_ham": "\u0664", "arap": "4", "latin": "dört", "tur": "num"},
            # Noktalama / sembol kaydı
            {"id": "wikt:sym", "arap_ham": "...", "arap": "...", "latin": "nokta", "tur": "punct"},
            # Geçerli Arap harfli kelime
            {"id": "wikt:valid", "arap_ham": "\u0643\u062a\u0627\u0628", "arap": "کتاب", "latin": "kitap", "tur": "noun"},
        ]
        kept, excluded, stats = clean_wiktionary(records)
        assert len(kept) == 1
        assert kept[0]["id"] == "wikt:valid"
        assert len(excluded) == 3
        assert stats["nedenler"]["arap_harfi_yok"] == 3
        assert all(ex["neden"] == "arap_harfi_yok" for ex in excluded)

    def test_drops_all_ten_digit_records(self) -> None:
        raw_digits = [
            ("\u06f0", "sıfr"), ("\u06f1", "bir"), ("\u06f2", "iki"), ("\u06f3", "üç"),
            ("\u0664", "dört"), ("\u0665", "beş"), ("\u0666", "altı"),
            ("\u06f7", "yedi"), ("\u06f8", "sekiz"), ("\u06f9", "dokuz")
        ]
        records = [
            {"id": f"wikt:digit:{i}", "arap_ham": char, "latin": lat, "tur": "num"}
            for i, (char, lat) in enumerate(raw_digits)
        ]
        kept, excluded, stats = clean_wiktionary(records)
        assert len(kept) == 0
        assert len(excluded) == 10
        assert stats["nedenler"]["arap_harfi_yok"] == 10
        assert all(ex["neden"] == "arap_harfi_yok" for ex in excluded)

    def test_wiktionary_multiple_exclusion_reasons(self) -> None:
        records = [
            # arap_harfi_yok
            {"id": "wikt:1", "arap_ham": "\u06f0", "latin": "sıfr", "tur": "num"},
            # latin_yok
            {"id": "wikt:2", "arap_ham": "\u0643\u062a\u0627\u0628", "latin": "", "tur": "noun"},
            # geçerli
            {"id": "wikt:3", "arap_ham": "\u0642\u0644\u0645", "latin": "kalem", "tur": "noun"},
        ]
        kept, excluded, stats = clean_wiktionary(records)
        assert len(kept) == 1
        assert len(excluded) == 2
        reasons = {ex["id"]: ex["neden"] for ex in excluded}
        assert reasons["wikt:1"] == "arap_harfi_yok"
        assert reasons["wikt:2"] == "latin_yok"
        assert stats["nedenler"]["arap_harfi_yok"] == 1
        assert stats["nedenler"]["latin_yok"] == 1


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

    def test_tie_breaker_fine_skeleton_b_p(self) -> None:
        """En yüksek iki adayın skor farkı <= 0.02 ise B/P ayrımıyla doğru aday seçilir."""
        # "\u062d\u0627\u067e\u0633\u0645\u0627\u0646\u0647" (حاپسمانه):
        # Adaylar: ["Habsamana", "Hapsamana"] -> kaba skorlar eşit (1.0),
        # پ harfi P olduğundan Hapsamana seçilmeli.
        records = [
            {
                "id": "gaz:tie:1",
                "arap_ham": "\u062d\u0627\u067e\u0633\u0645\u0627\u0646\u0647",
                "latin": "Habsamana / Hapsamana",
                "tur": "yer_adi",
            }
        ]
        kept, excluded, stats = clean_gazetteer(records)
        assert len(kept) == 1
        rec = kept[0]
        assert rec["latin"] == "Hapsamana"
        assert rec["eslesme_skoru"] == 1.0
        assert rec["guvenilir"] is True
        assert rec["latin_adaylar"] == ["Habsamana", "Hapsamana"]

    def test_tie_breaker_fine_skeleton_b_p_reverse_order(self) -> None:
        """Adayların sırası ters olduğunda da doğru aday (Hapsamana) seçilir."""
        records = [
            {
                "id": "gaz:tie:2",
                "arap_ham": "\u062d\u0627\u067e\u0633\u0645\u0627\u0646\u0647",
                "latin": "Hapsamana / Habsamana",
                "tur": "yer_adi",
            }
        ]
        kept, excluded, stats = clean_gazetteer(records)
        assert len(kept) == 1
        assert kept[0]["latin"] == "Hapsamana"

    def test_tie_breaker_fine_skeleton_c_cedilla(self) -> None:
        """En yüksek iki adayın skor farkı <= 0.02 ise C/Ç ayrımıyla doğru aday seçilir."""
        # "\u0686\u0634\u0645\u0647" (çeşme):
        # Adaylar: ["Cesme", "Çeşme"] -> kaba skorlar eşit (1.0),
        # چ harfi Ç olduğundan Çeşme seçilmeli.
        records = [
            {
                "id": "gaz:tie:3",
                "arap_ham": "\u0686\u0634\u0645\u0647",
                "latin": "Cesme / Çeşme",
                "tur": "yer_adi",
            }
        ]
        kept, excluded, stats = clean_gazetteer(records)
        assert len(kept) == 1
        assert kept[0]["latin"] == "Çeşme"
        assert kept[0]["eslesme_skoru"] == 1.0

    def test_tie_breaker_diff_greater_than_002_prefers_coarse(self) -> None:
        """Skor farkı > 0.02 olduğunda ince iskelet devreye girmez, kaba skoru yüksek aday seçilir."""
        records = [
            {
                "id": "gaz:tie:4",
                "arap_ham": "\u0642\u0627\u0646\u0644\u06cc \u0627\u0637\u0647",
                "latin": "Kanlıada / İpsili",
                "tur": "ada",
            }
        ]
        kept, excluded, stats = clean_gazetteer(records)
        assert len(kept) == 1
        # Kanlıada skoru ~1.0, İpsili skoru ~0.3 (fark > 0.02)
        assert kept[0]["latin"] == "Kanlıada"

    def test_tie_breaker_equal_fine_scores_preserves_order(self) -> None:
        """Kaba ve ince iskelet skorları eşit olduğunda ilk sıradaki aday korunur."""
        records = [
            {
                "id": "gaz:tie:5",
                "arap_ham": "\u0642\u0644\u0645",
                "latin": "Kalem / Kalam",
                "tur": "sozcuk",
            }
        ]
        kept, excluded, stats = clean_gazetteer(records)
        assert len(kept) == 1
        assert kept[0]["latin"] == "Kalem"

    def test_tie_breaker_single_candidate(self) -> None:
        """Tek adaylı kayıtlarda eşitlik bozma aranmaz, doğrudan aday seçilir."""
        records = [
            {
                "id": "gaz:single:1",
                "arap_ham": "\u0642\u0644\u0645",
                "latin": "Kalem",
                "tur": "sozcuk",
            }
        ]
        kept, excluded, stats = clean_gazetteer(records)
        assert len(kept) == 1
        assert kept[0]["latin"] == "Kalem"
