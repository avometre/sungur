"""Osmanlıca ve Latin metinleri için ünsüz iskeleti benzerliği modülü.

Bu modül; Arap harfli Osmanlıca metin ile Latin transkripsiyonunun kabaca
aynı sözcük veya cümleyi temsil edip etmediğini ölçmek için ünlü harfleri eleyerek
ve benzer ünsüzleri ortak sınıflara indirgeyerek iskelet benzerliği (similarity)
hesaplar.
"""

import difflib
import regex

from src.common.normalize import normalize_latin, normalize_ottoman, tr_lower

# Arapça Harf Sınıfı Eşlemeleri (Çıktı sınıfları)
# ب پ → B      ت ط ة د → T      ث س ص → S      ج چ → C      ح خ ه ۀ → H
# ذ ز ض ظ → Z  ر → R            ژ → J          ش → X       غ ق ک گ → K
# ڭ ن → N      ف → F            ل → L          م → M
ARABIC_CONSONANT_MAP: dict[str, str] = {
    # ب پ → B
    "\u0628": "B",  # ب
    "\u067E": "B",  # پ
    # ت ط ة د → T
    "\u062A": "T",  # ت
    "\u0637": "T",  # ط
    "\u0629": "T",  # ة
    "\u062F": "T",  # د
    # ث س ص → S
    "\u062B": "S",  # ث
    "\u0633": "S",  # س
    "\u0635": "S",  # ص
    # ج چ → C
    "\u062C": "C",  # ج
    "\u0686": "C",  # چ
    # ح خ ه ۀ → H
    "\u062D": "H",  # ح
    "\u062E": "H",  # خ
    "\u0647": "H",  # ه
    "\u06C0": "H",  # ۀ
    # ذ ز ض ظ → Z
    "\u0630": "Z",  # ذ
    "\u0632": "Z",  # ز
    "\u0636": "Z",  # ض
    "\u0638": "Z",  # ظ
    # ر → R
    "\u0631": "R",  # ر
    # ژ → J
    "\u0698": "J",  # ژ
    # ش → X
    "\u0634": "X",  # ش
    # غ ق ک گ → K
    "\u063A": "K",  # غ
    "\u0642": "K",  # ق
    "\u06A9": "K",  # ک
    "\u0643": "K",  # ك (normalize_ottoman kaçırırsa yedek)
    "\u06AF": "K",  # گ
    # ڭ ن → N
    "\u06AD": "N",  # ڭ
    "\u0763": "N",  # ݣ
    "\u0646": "N",  # ن
    # ف → F
    "\u0641": "F",  # ف
    # ل → L
    "\u0644": "L",  # ل
    # م → M
    "\u0645": "M",  # م
}

# Latin Harf Sınıfı Eşlemeleri
# b p → B   t d → T   s → S   c ç → C   h → H   z → Z   r → R   j → J   ş → X
# k g ğ q → K   n ñ → N   f → F   l → L   m → M
LATIN_CONSONANT_MAP: dict[str, str] = {
    "b": "B",
    "p": "B",
    "t": "T",
    "d": "T",
    "s": "S",
    "c": "C",
    "ç": "C",
    "h": "H",
    "z": "Z",
    "r": "R",
    "j": "J",
    "ş": "X",
    "k": "K",
    "g": "K",
    "ğ": "K",
    "q": "K",
    "n": "N",
    "ñ": "N",
    "f": "F",
    "l": "L",
    "m": "M",
}

# Kelime çıkarma ve sondaki harf dışı ekleri temizleme desenleri
ARABIC_WORD_PATTERN: regex.Pattern = regex.compile(r"[\p{Arabic}\u200C]+")
TRAILING_NON_LETTER_PATTERN: regex.Pattern = regex.compile(r"[\p{M}\u200C\u0640]+$")


def arabic_skeleton(arap: str) -> str:
    """Arap harfli metnin ünsüz iskeletini çıkarır.

    1. normalize_ottoman ile standartlaştırılır.
    2. Kelimelere bölünür; her kelimenin sonundaki ه veya ۀ atılır (e/a ünlüsü).
    3. Ünlü taşıyıcıları (ا آ أ إ ع ء ئ ؤ و ی), harekeler, ZWNJ, boşluk, rakam
       ve eşleme tablosunda bulunmayan tüm işaretler elenir.
    4. Kalan ünsüzler büyük harf sınıflarına (B, T, S, C, H, Z, R, J, X, K, N, F, L, M) dönüştürülür.
    """
    if not arap:
        return ""

    norm = normalize_ottoman(arap)
    words = ARABIC_WORD_PATTERN.findall(norm)
    skeleton_chars: list[str] = []

    for w in words:
        # Kelime sonundaki hareke, tatvil ve zwnj karakterlerini temizle
        w = TRAILING_NON_LETTER_PATTERN.sub("", w)
        # Kelime sonundaki ه veya ۀ harfini at
        if w.endswith(("\u0647", "\u06C0")):
            w = w[:-1]

        for ch in w:
            mapped = ARABIC_CONSONANT_MAP.get(ch)
            if mapped:
                skeleton_chars.append(mapped)

    return "".join(skeleton_chars)


def latin_skeleton(latin: str) -> str:
    """Latin harfli metnin ünsüz iskeletini çıkarır.

    1. tr_lower + normalize_latin(strip_diacritics=True) uygulanır.
    2. Ünlüler (a e ı i o ö u ü â î û v y) ve harf olmayan tüm karakterler elenir.
    3. Konsonantlar büyük harf sınıflarına (B, T, S, C, H, Z, R, J, X, K, N, F, L, M) dönüştürülür.
    """
    if not latin:
        return ""

    s = tr_lower(latin)
    s = normalize_latin(s, strip_diacritics=True)
    skeleton_chars: list[str] = []

    for ch in s:
        mapped = LATIN_CONSONANT_MAP.get(ch)
        if mapped:
            skeleton_chars.append(mapped)

    return "".join(skeleton_chars)


def skeleton_similarity(arap: str, latin: str) -> float:
    """Arap harfli metin ile Latin metin arasındaki ünsüz iskeleti benzerliğini [0, 1] aralığında döner.

    Her iki iskelet de boşsa 0.0 döner.
    """
    s_a = arabic_skeleton(arap)
    s_l = latin_skeleton(latin)

    if not s_a and not s_l:
        return 0.0

    return difflib.SequenceMatcher(None, s_a, s_l).ratio()
