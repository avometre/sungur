"""Osmanlıca ve transkripsiyonlu metinler için normalizasyon modülü.

Bu modül; Osmanlı Türkçesi metinlerinin Arap harfli imlasını standartlaştırmak,
sunum formlarını çözmek, transkripsiyonlu Latin alfabelerini sadeleştirmek ve
Türkçe kurallarına uygun küçük harf ile Arapça oran hesaplamalarını yapmak için
kullanılır.
"""

import unicodedata
import regex

# ---------------------------------------------------------------------------
# Sabit Tanımları (Constants)
# ---------------------------------------------------------------------------

# 1. Presentation Forms (Sunum Formları) aralıkları:
# Arabic Presentation Forms-A (U+FB50-U+FDFF) ve Forms-B (U+FE70-U+FEFC)
PRESENTATION_FORMS_PATTERN: regex.Pattern = regex.compile(
    r"[\uFB50-\uFDFF\uFE70-\uFEFC]"
)

# 2. Yön Kontrol Karakterleri (Directional Controls to strip)
DIRECTIONAL_CHARS: set[str] = {
    "\u200E",  # LEFT-TO-RIGHT MARK (LRM)
    "\u200F",  # RIGHT-TO-LEFT MARK (RLM)
    "\u202A",  # LEFT-TO-RIGHT EMBEDDING (LRE)
    "\u202B",  # RIGHT-TO-LEFT EMBEDDING (RLE)
    "\u202C",  # POP DIRECTIONAL FORMATTING (PDF)
    "\u202D",  # LEFT-TO-RIGHT OVERRIDE (LRO)
    "\u202E",  # RIGHT-TO-LEFT OVERRIDE (RLO)
    "\u2066",  # LEFT-TO-RIGHT ISOLATE (LRI)
    "\u2067",  # RIGHT-TO-LEFT ISOLATE (RLI)
    "\u2068",  # FIRST STRONG ISOLATE (FSI)
    "\u2069",  # POP DIRECTIONAL ISOLATE (PDI)
    "\uFEFF",  # ZERO WIDTH NO-BREAK SPACE / BYTE ORDER MARK (BOM)
}

# 3. Sıfır Genişlikli Karakterler (Silinecekler)
# U+200C (ZWNJ) Osmanlıca ek ayrımı için KORUNUR (örn. پادشاه‌لر).
ZERO_WIDTH_CHARS_TO_REMOVE: set[str] = {
    "\u200B",  # ZERO WIDTH SPACE (ZWSP)
    "\u200D",  # ZERO WIDTH JOINER (ZWJ)
}

# 4. Tatvil / Keşide (Kashida / Tatweel)
TATWEEL_CHAR: str = "\u0640"

# 5. Harekeler (U+064B–U+065F ve U+0670 Üstteki Küçük Elif)
HARAKAT_CHARS: set[str] = {
    chr(cp) for cp in range(0x064B, 0x065F + 1)
} | {"\u0670"}

# 6. Harf Birleştirme Eşlemeleri (Letter Unification)
# - ي (U+064A) -> ی (U+06CC)
# - ى (U+0649) -> ی (U+06CC)
# - ك (U+0643) -> ک (U+06A9)
# - ہ (U+06C1) -> ه (U+0647)
LETTER_UNIFICATION_MAP: dict[str, str] = {
    "\u064A": "\u06CC",  # ي (Arabic Yeh) -> ی (Farsi Yeh)
    "\u0649": "\u06CC",  # ى (Alef Maksura) -> ی (Farsi Yeh)
    "\u0643": "\u06A9",  # ك (Arabic Kaf) -> ک (Keheh)
    "\u06C1": "\u0647",  # ہ (Urdu Heh Goal) -> ه (Arabic Heh)
}

# Korunacak Osmanlıca Harfler (Referans / Güvence amaçlı küme)
PROTECTED_OTTOMAN_LETTERS: set[str] = {
    "\u06AD",  # ڭ (Nef / Sağır Kef)
    "\u06AF",  # گ (Gef)
    "\u0698",  # ژ (Zhe)
    "\u0686",  # چ (Tche)
    "\u067E",  # پ (Peh)
    "\u0624",  # ؤ (Waw with Hamza Above)
    "\u0626",  # ئ (Yeh with Hamza Above)
    "\u0622",  # آ (Alef with Madda Above)
    "\u0629",  # ة (Teh Marbuta)
    "\u06CB",  # ۋ (Ve / Waw with Dot Above)
}

# 7. Hemze Katlama (Fold Hamza)
HAMZA_FOLDING_MAP: dict[str, str] = {
    "\u0671": "\u0627",  # ٱ (Alef Wasla) -> ا
    "\u0625": "\u0627",  # إ (Alef with Hamza Below) -> ا
    "\u0623": "\u0627",  # أ (Alef with Hamza Above) -> ا
}

# 8. Rakam Eşlemeleri (Arap-Hint ve Farsça -> ASCII 0-9)
DIGIT_TRANSLATION_MAP: dict[str, str] = {
    # Arap-Hint (U+0660–0669)
    "\u0660": "0",
    "\u0661": "1",
    "\u0662": "2",
    "\u0663": "3",
    "\u0664": "4",
    "\u0665": "5",
    "\u0666": "6",
    "\u0667": "7",
    "\u0668": "8",
    "\u0669": "9",
    # Doğu Arap-Hint / Farsça (U+06F0–06F9)
    "\u06F0": "0",
    "\u06F1": "1",
    "\u06F2": "2",
    "\u06F3": "3",
    "\u06F4": "4",
    "\u06F5": "5",
    "\u06F6": "6",
    "\u06F7": "7",
    "\u06F8": "8",
    "\u06F9": "9",
}

# 9. Noktalama İşaretleri Eşlemesi
PUNCTUATION_MAP: dict[str, str] = {
    "\u060C": ",",  # ، (Arabic Comma)
    "\u061F": "?",  # ؟ (Arabic Question Mark)
    "\u061B": ";",  # ؛ (Arabic Semicolon)
}

# 10. Boşluk Deseni
WHITESPACE_PATTERN: regex.Pattern = regex.compile(r"[\s\u00A0]+")

# Latin Normalizasyon Sabitleri
QUOTE_MAP: dict[str, str] = {
    "’": "'",
    "‘": "'",
    "`": "'",
    "´": "'",
    "ʹ": "'",  # U+02B9
}

AYN_HAMZA_MAP: dict[str, str] = {
    "\u02BB": "\u02BF",  # ʻ (Modifier Letter Turned Comma) -> ʿ
    "\u02BC": "\u02BE",  # ʼ (Modifier Letter Apostrophe) -> ʾ
}

# Transkripsiyon harflerinin düzleştirme eşleme tablosu
# Türkçe karakterler (ç ğ ı İ ö ş ü Ç Ğ Ö Ş Ü) KORUNUR!
TRANSCRIPTION_DIACRITICS_MAP: dict[str, str] = {
    "â": "a",
    "Â": "A",
    "î": "i",
    "Î": "I",
    "û": "u",
    "Û": "U",
    "ā": "a",
    "Ā": "A",
    "ī": "i",
    "Ī": "I",
    "ū": "u",
    "Ū": "U",
    "ḥ": "h",
    "Ḥ": "H",
    "ḫ": "h",
    "Ḫ": "H",
    "ṣ": "s",
    "Ṣ": "S",
    "ṭ": "t",
    "Ṭ": "T",
    "ż": "z",
    "Ż": "Z",
    "ẓ": "z",
    "Ẓ": "Z",
    "ḍ": "d",
    "Ḍ": "D",
    "ẕ": "z",
    "Ẕ": "Z",
    "ŝ": "s",
    "Ŝ": "S",
    "ñ": "n",
    "Ñ": "N",
    "ġ": "g",
    "Ġ": "G",
    "ḳ": "k",
    "Ḳ": "K",
    "ṯ": "t",
    "Ṯ": "T",
    "ḏ": "d",
    "Ḏ": "D",
    "ʿ": "",  # U+02BF silinir
    "ʾ": "",  # U+02BE silinir
}


# ---------------------------------------------------------------------------
# Normalizasyon Fonksiyonları
# ---------------------------------------------------------------------------


def normalize_ottoman(
    text: str,
    keep_harakat: bool = False,
    fold_hamza: bool = False,
    ascii_digits: bool = True,
) -> str:
    """Osmanlıca metinleri standartlaştırmak için normalizasyon adımlarını uygular.

    Sırayla:
    1. NFC normalizasyonu + sunum formlarını (U+FB50–U+FDFF, U+FE70–U+FEFC) NFKC ile çözme
    2. Yön kontrol karakterlerini silme
    3. Sıfır genişlikli karakterleri (U+200B, U+200D) silme (U+200C / ZWNJ korunur)
    4. Tatvil (U+0640) silme
    5. keep_harakat=False ise harekeleri silme (U+064B–U+065F ve U+0670)
    6. Harf birleştirme (ي, ى -> ی, ك -> ک, ہ -> ه)
    7. fold_hamza=True ise ٱ, إ, أ -> ا dönüştürme
    8. ascii_digits=True ise Arap-Hint ve Farsça rakamları 0-9'a çevirme
    9. Arapça noktalama işaretlerini (, ? ;) çevirme
    10. Tüm boşlukları tek boşluğa indirip baştan/sondan kırpma
    """
    if not text:
        return ""

    # 1. NFC normalizasyonu
    text = unicodedata.normalize("NFC", text)

    # 1.b. Sadece sunum formlarına NFKC uygula
    text = PRESENTATION_FORMS_PATTERN.sub(
        lambda m: unicodedata.normalize("NFKC", m.group(0)), text
    )

    result_chars: list[str] = []

    for ch in text:
        # 2. Yön kontrol karakterlerini sil
        if ch in DIRECTIONAL_CHARS:
            continue

        # 3. Sıfır genişlikli karakterleri sil (ZWNJ U+200C korunur)
        if ch in ZERO_WIDTH_CHARS_TO_REMOVE:
            continue

        # 4. Tatvil (U+0640) sil
        if ch == TATWEEL_CHAR:
            continue

        # 5. Harekeleri sil
        if not keep_harakat and ch in HARAKAT_CHARS:
            continue

        # 6. Harf birleştirme
        if ch in LETTER_UNIFICATION_MAP:
            ch = LETTER_UNIFICATION_MAP[ch]

        # 7. fold_hamza=True ise hemzeleri katla
        if fold_hamza and ch in HAMZA_FOLDING_MAP:
            ch = HAMZA_FOLDING_MAP[ch]

        # 8. Rakam dönüştürme
        if ascii_digits and ch in DIGIT_TRANSLATION_MAP:
            ch = DIGIT_TRANSLATION_MAP[ch]

        # 9. Noktalama dönüştürme
        if ch in PUNCTUATION_MAP:
            ch = PUNCTUATION_MAP[ch]

        result_chars.append(ch)

    # 10. Boşlukları tek boşluğa indir ve kırp
    normalized = "".join(result_chars)
    return WHITESPACE_PATTERN.sub(" ", normalized).strip()


def normalize_latin(text: str, strip_diacritics: bool = False) -> str:
    """Transkripsiyonlu Latin alfabesi metinlerini normalize eder.

    Sırayla:
    1. unicodedata.normalize("NFC", text)
    2. Kesme/tırnak birleştirme: ’ ‘ ` ´ ʹ → '
    3. Ayn ve hemze işaretleri: ʿ (U+02BF) ve ʾ (U+02BE) korunur;
       ʻ (U+02BB) → ʿ ve ʼ (U+02BC) → ʾ dönüştürülür.
    4. strip_diacritics=True ise transkripsiyon işaretleri kaldırılır,
       fakat Türkçe harfler (ç ğ ı İ ö ş ü Ç Ğ Ö Ş Ü) kesinlikle KORUNUR.
    5. Boşlukları tek boşluğa indirip kırpma.
    """
    if not text:
        return ""

    # 1. NFC normalizasyonu
    text = unicodedata.normalize("NFC", text)

    result_chars: list[str] = []

    for ch in text:
        # 2. Kesme / tırnak birleştirme
        if ch in QUOTE_MAP:
            ch = QUOTE_MAP[ch]

        # 3. Ayn ve hemze işaret standardizasyonu
        if ch in AYN_HAMZA_MAP:
            ch = AYN_HAMZA_MAP[ch]

        # 4. strip_diacritics=True ise transkripsiyon işaretlerini kaldır
        if strip_diacritics:
            if ch in TRANSCRIPTION_DIACRITICS_MAP:
                ch = TRANSCRIPTION_DIACRITICS_MAP[ch]

        result_chars.append(ch)

    # 5. Boşlukları tek boşluğa indir ve kırp
    normalized = "".join(result_chars)
    return WHITESPACE_PATTERN.sub(" ", normalized).strip()


def tr_lower(text: str) -> str:
    """Türkçe kurallarına uygun küçük harf dönüşümü yapar.

    'I' -> 'ı' ve 'İ' -> 'i' dönüşümü yapıldıktan sonra .lower() çağrılır.
    """
    if not text:
        return ""
    return text.replace("İ", "i").replace("I", "ı").lower()


def arabic_ratio(text: str) -> float:
    """Boşluk dışı karakterler içinde Arap betiği karakterlerinin oranını (\\p{Arabic}) hesaplar.

    Boş metin veya sadece boşluk içeren metin durumunda 0.0 döner.
    """
    non_space_chars = [c for c in text if not c.isspace()]
    if not non_space_chars:
        return 0.0

    joined = "".join(non_space_chars)
    arabic_chars = regex.findall(r"\p{Arabic}", joined)
    return len(arabic_chars) / len(non_space_chars)
