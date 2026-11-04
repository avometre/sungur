"""Vikikaynak satır hizalama (align_lines) ve HTML temizleme testleri."""

from src.scrapers.vikikaynak_align import align_lines, clean_latin_html


class TestAlignLines:
    def test_three_identical_lines(self) -> None:
        # 3 Arapça + 3 Latin özdeş sırada → üç adet 1-1
        # "\u06a9\u062a\u0627\u0628" (kitâb), "\u062f\u0648\u0644\u062a \u0639\u0644\u06cc\u0647" (devlet-i aliyye), "\u067e\u0627\u062f\u0634\u0627\u0647" (pâdişâh)
        arap_lines = [
            "\u06a9\u062a\u0627\u0628",
            "\u062f\u0648\u0644\u062a \u0639\u0644\u06cc\u0647",
            "\u067e\u0627\u062f\u0634\u0627\u0647",
        ]
        latin_lines = ["kitap", "devlet-i aliyye", "padişah"]

        pairs, stats = align_lines(arap_lines, latin_lines)
        assert len(pairs) == 3
        assert [p["hizalama"] for p in pairs] == ["1-1", "1-1", "1-1"]
        assert all(p["skor"] >= 0.8 for p in pairs)
        assert stats["atlanan_arap"] == 0
        assert stats["atlanan_latin"] == 0

    def test_extra_title_line_in_latin_skipped(self) -> None:
        # Latin tarafına fazladan bir başlık satırı eklenmiş → o satır atlanıyor, kalanlar doğru eşleşiyor
        # "\u06a9\u062a\u0627\u0628" (kitâb), "\u062f\u0648\u0644\u062a \u0639\u0644\u06cc\u0647" (devlet-i aliyye)
        arap_lines = ["\u06a9\u062a\u0627\u0628", "\u062f\u0648\u0644\u062a \u0639\u0644\u06cc\u0647"]
        latin_lines = ["Bölüm I: Giriş ve Başlık", "kitap", "devlet-i aliyye"]

        pairs, stats = align_lines(arap_lines, latin_lines)
        assert len(pairs) == 2
        assert [p["hizalama"] for p in pairs] == ["1-1", "1-1"]
        assert pairs[0]["latin"] == "kitap"
        assert pairs[1]["latin"] == "devlet-i aliyye"
        assert stats["atlanan_latin"] == 1
        assert stats["atlanan_arap"] == 0

    def test_one_arabic_split_into_two_latin(self) -> None:
        # Bir Arapça satırın Latin'de iki satıra bölünmüş olduğu durum → 1-2
        # "\u06a9\u062a\u0627\u0628 \u067e\u0627\u062f\u0634\u0627\u0647" (kitâb-ı pâdişâh)
        arap_lines = ["\u06a9\u062a\u0627\u0628 \u067e\u0627\u062f\u0634\u0627\u0647"]
        latin_lines = ["kitap", "padişah"]

        pairs, stats = align_lines(arap_lines, latin_lines)
        assert len(pairs) == 1
        assert pairs[0]["hizalama"] == "1-2"
        assert pairs[0]["latin"] == "kitap padişah"
        assert pairs[0]["skor"] >= 0.8
        assert stats["atlanan_arap"] == 0
        assert stats["atlanan_latin"] == 0

    def test_two_arabic_merged_into_one_latin(self) -> None:
        # İki Arapça satırın Latin'de tek satırda birleştiği durum → 2-1
        # "\u0627\u0633\u062a\u0627\u0646\u0628\u0648\u0644" (İstanbul), "\u067e\u0627\u062f\u0634\u0627\u0647" (pâdişâh)
        arap_lines = ["\u0627\u0633\u062a\u0627\u0646\u0628\u0648\u0644", "\u067e\u0627\u062f\u0634\u0627\u0647"]
        latin_lines = ["İstanbul padişah"]

        pairs, stats = align_lines(arap_lines, latin_lines)
        assert len(pairs) == 1
        assert pairs[0]["hizalama"] == "2-1"
        assert pairs[0]["skor"] >= 0.8

    def test_empty_lines_input(self) -> None:
        pairs, stats = align_lines([], ["kitap"])
        assert pairs == []
        assert stats["atlanan_latin"] == 1
        assert stats["atlanan_arap"] == 0


class TestCleanLatinHtml:
    def test_clean_html_removes_unwanted_tags(self) -> None:
        html = """
        <div class="headertemplate">Gereksiz Başlık</div>
        <div class="mw-parser-output">
            <p>Birinci satır metin burada.<br>İkinci dize burada başlıyor.</p>
            <div class="navbox">Navigasyon kutusu</div>
            <div class="noprint">Çokdilli Vikikaynak bağlantısı</div>
            <p>Üçüncü satır devam ediyor.</p>
        </div>
        """
        lines = clean_latin_html(html)
        assert len(lines) == 3
        assert "Birinci satır metin burada." in lines
        assert "İkinci dize burada başlıyor." in lines
        assert "Üçüncü satır devam ediyor." in lines
        assert not any("Navigasyon" in l for l in lines)
        assert not any("Gereksiz" in l for l in lines)
        assert not any("Çokdilli" in l for l in lines)
