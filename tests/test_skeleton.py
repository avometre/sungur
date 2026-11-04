"""Ünsüz iskeleti ve benzerlik modülü testleri."""

import pytest
from src.translit.skeleton import (
    arabic_skeleton,
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
