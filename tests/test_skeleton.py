"""Ünsüz iskeleti ve benzerlik modülü testleri."""

import pytest
from src.translit.skeleton import (
    arabic_skeleton,
    fine_skeleton,
    fine_skeleton_similarity,
    latin_skeleton,
    skeleton_similarity,
)


class TestSkeleton:
    def test_arabic_skeleton_basic(self) -> None:
        # "\u06a9\u062a\u0627\u0628" (kitâb) -> KTB
        assert arabic_skeleton("\u06a9\u062a\u0627\u0628") == "KTB"
        # "\u067e\u0627\u062f\u0634\u0627\u0647" (pâdişâh): son ه atılır -> BTX
        assert arabic_skeleton("\u067e\u0627\u062f\u0634\u0627\u0647") == "BTX"

    def test_arabic_skeleton_strips_final_heh(self) -> None:
        # "\u062e\u06cc\u0631\u0647 \u0628\u0648\u0644\u06cc" (Hayrabolu): خیره sonundaki ه atılır -> HR, بولی -> BL
        assert arabic_skeleton("\u062e\u06cc\u0631\u0647 \u0628\u0648\u0644\u06cc") == "HRBL"
        # "\u062e\u0627\u0646\u06c0" (hâne-i): ۀ atılır -> HN
        assert arabic_skeleton("\u062e\u0627\u0646\u06c0") == "HN"

    def test_latin_skeleton_basic(self) -> None:
        assert latin_skeleton("kitap") == "KTB"
        assert latin_skeleton("Hayrabolu") == "HRBL"
        assert latin_skeleton("devlet-i aliyye") == "TLTL"
        assert latin_skeleton("padişah") == "BTXH"

    def test_empty_inputs(self) -> None:
        assert arabic_skeleton("") == ""
        assert latin_skeleton("") == ""
        assert skeleton_similarity("", "") == 0.0
        assert skeleton_similarity("   ", "   ") == 0.0
        assert skeleton_similarity("\u0627\u0648\u06cc", "aeiou") == 0.0

    def test_required_similarity_pairs(self) -> None:
        # 1. "\u06a9\u062a\u0627\u0628" (kitâb) ~ "kitap" >= 0.8
        sim_kitap = skeleton_similarity("\u06a9\u062a\u0627\u0628", "kitap")
        assert sim_kitap >= 0.8

        # 2. "\u062e\u06cc\u0631\u0647 \u0628\u0648\u0644\u06cc" (Hayrabolu) ~ "Hayrabolu" >= 0.8
        sim_hayrabolu = skeleton_similarity(
            "\u062e\u06cc\u0631\u0647 \u0628\u0648\u0644\u06cc", "Hayrabolu"
        )
        assert sim_hayrabolu >= 0.8

        # 3. "\u0642\u0627\u0646\u0644\u06cc \u0627\u0637\u0647" (Kanlıada) ~ "Kanlıada" >= 0.8
        sim_kanliada = skeleton_similarity(
            "\u0642\u0627\u0646\u0644\u06cc \u0627\u0637\u0647", "Kanlıada"
        )
        assert sim_kanliada >= 0.8

        # 4. "\u0642\u0627\u0646\u0644\u06cc \u0627\u0637\u0647" (Kanlıada) ~ "İpsili" <= 0.4
        sim_ipsili = skeleton_similarity(
            "\u0642\u0627\u0646\u0644\u06cc \u0627\u0637\u0647", "İpsili"
        )
        assert sim_ipsili <= 0.4

        # 5. "\u062f\u0648\u0644\u062a \u0639\u0644\u06cc\u0647" (devlet-i aliyye) ~ "devlet-i aliyye" >= 0.8
        sim_devlet = skeleton_similarity(
            "\u062f\u0648\u0644\u062a \u0639\u0644\u06cc\u0647", "devlet-i aliyye"
        )
        assert sim_devlet >= 0.8


class TestFineSkeleton:
    """İnce ünsüz iskeleti (B vs P, C vs Ç) birim testleri."""

    def test_fine_skeleton_b_vs_p_arabic(self) -> None:
        assert fine_skeleton("\u0628\u0627\u0632\u0627\u0631", is_arabic=True) == "BZR"
        assert fine_skeleton("\u067e\u0627\u0632\u0627\u0631", is_arabic=True) == "PZR"
        assert arabic_skeleton("\u0628\u0627\u0632\u0627\u0631") == "BZR"
        assert arabic_skeleton("\u067e\u0627\u0632\u0627\u0631") == "BZR"

    def test_fine_skeleton_b_vs_p_latin(self) -> None:
        assert fine_skeleton("bazar", is_arabic=False) == "BZR"
        assert fine_skeleton("pazar", is_arabic=False) == "PZR"
        assert latin_skeleton("bazar") == "BZR"
        assert latin_skeleton("pazar") == "BZR"

    def test_fine_skeleton_c_vs_c_cedilla_arabic(self) -> None:
        assert fine_skeleton("\u062c\u0627\u0645", is_arabic=True) == "CM"
        assert fine_skeleton("\u0686\u0627\u0645", is_arabic=True) == "ÇM"
        assert fine_skeleton("\u0686\u0627\u06cc", is_arabic=True) == "Ç"
        assert arabic_skeleton("\u062c\u0627\u0645") == "CM"
        assert arabic_skeleton("\u0686\u0627\u0645") == "CM"
        assert arabic_skeleton("\u0686\u0627\u06cc") == "C"

    def test_fine_skeleton_c_vs_c_cedilla_latin(self) -> None:
        assert fine_skeleton("cam", is_arabic=False) == "CM"
        assert fine_skeleton("çam", is_arabic=False) == "ÇM"
        assert fine_skeleton("çay", is_arabic=False) == "Ç"
        assert fine_skeleton("cay", is_arabic=False) == "C"
        assert latin_skeleton("cam") == "CM"
        assert latin_skeleton("çam") == "CM"
        assert latin_skeleton("çay") == "C"
        assert latin_skeleton("cay") == "C"

    def test_fine_skeleton_auto_detect_script(self) -> None:
        assert fine_skeleton("\u06a9\u062a\u0627\u0628") == "KTB"
        assert fine_skeleton("\u067e\u0627\u0632\u0627\u0631") == "PZR"
        assert fine_skeleton("\u0686\u0627\u06cc") == "Ç"
        assert fine_skeleton("kitap") == "KTP"
        assert fine_skeleton("pazar") == "PZR"
        assert fine_skeleton("çay") == "Ç"

    def test_fine_skeleton_empty_and_whitespace(self) -> None:
        assert fine_skeleton("") == ""
        assert fine_skeleton("   ") == ""
        assert fine_skeleton_similarity("", "") == 0.0
        assert fine_skeleton_similarity("   ", "   ") == 0.0
        assert fine_skeleton_similarity("\u0627\u0648\u06cc", "aeiou") == 0.0

    def test_fine_similarity_habsamana_hapsamana(self) -> None:
        arap = "\u062d\u0627\u067e\u0633\u0645\u0627\u0646\u0647"
        coarse_habsamana = skeleton_similarity(arap, "Habsamana")
        coarse_hapsamana = skeleton_similarity(arap, "Hapsamana")
        assert coarse_habsamana == 1.0
        assert coarse_hapsamana == 1.0
        assert abs(coarse_habsamana - coarse_hapsamana) <= 0.02

        fine_habsamana = fine_skeleton_similarity(arap, "Habsamana")
        fine_hapsamana = fine_skeleton_similarity(arap, "Hapsamana")
        assert fine_hapsamana == 1.0
        assert fine_habsamana == 0.8
        assert fine_hapsamana > fine_habsamana

    def test_fine_similarity_cay_vs_cay_cedilla(self) -> None:
        arap = "\u0686\u0627\u06cc"
        assert skeleton_similarity(arap, "çay") == 1.0
        assert skeleton_similarity(arap, "cay") == 1.0
        assert fine_skeleton_similarity(arap, "çay") == 1.0
        assert fine_skeleton_similarity(arap, "cay") == 0.0
