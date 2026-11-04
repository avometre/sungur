"""Osmanlıca ve transkripsiyonlu metinler için normalizasyon modülü.

Bu modül; Osmanlı Türkçesi metinlerinin Arap harfli imlasını standartlaştırmak,
izafet hemzesini ve Osmanlıca harflerini korumak, sunum formlarını çözmek,
transkripsiyonlu Latin alfabelerini sadeleştirmek ve Türkçe kurallarına uygun
küçük harf ile Arapça oran hesaplamalarını yapmak için kullanılır.
"""

import unicodedata
import regex

# ---------------------------------------------------------------------------
# Sabit Tanımları (Constants)
# ---------------------------------------------------------------------------

# 1. Harf Sunum Formları (Arabic Presentation Forms-A ve Forms-B)
# Not: U+FE70–U+FE7F (tek başına hareke sunum formları) NFKC uygulandığında
# kelime ortasına istenmeyen boşluk eklediği için bu aralıktan çıkarılmıştır.
PRESENTATION_FORMS_PATTERN: regex.Pattern = regex.compile(
    r"[\uFB50-\uFDFF\uFE80-\uFEFC]"
)

# 1.b. Hareke Sunum Formları Eşleme Tablosu (U+FE70–U+FE7F)
# Modül yüklenirken NFKC sonucundan boşluk ve U+0640 (tatvil) silinerek otomatik üretilir.
# U+FE73 Arapça hareke olmadığı için atlanır.
HARAKAT_PRESENTATION_MAP: dict[str, str] = {}
for _cp in range(0xFE70, 0xFE80):
    if _cp == 0xFE73:
        continue
    _ch = chr(_cp)
    _decomp = (
        unicodedata.normalize("NFKC", _ch).replace(" ", "").replace("\u0640", "")
    )
    if _decomp:
        HARAKAT_PRESENTATION_MAP[_ch] = _decomp

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
# U+0654 ve U+0655 (üstte/altta birleşik hemze) izafet ve hemze korunumu için
# hareke kümesinden ÇIKARILMIŞTIR.
HARAKAT_CHARS: set[str] = (
    {chr(cp) for cp in range(0x064B, 0x065F + 1)} - {"\u0654", "\u0655"}
) | {"\u0670"}

# 6. Harf + Birleşik Hemze Birleştirme Eşlemeleri
# ه + U+0654 -> ۀ (U+06C0, hâne-i izafeti)
# ی/ي + U+0654 -> ئ (U+0626)
# و + U+0654 -> ؤ (U+0624)
# ا + U+0654 -> أ (U+0623)
# ا + U+0655 -> إ (U+0625)
HAMZA_BASE_MAP: dict[tuple[str, str], str] = {
    ("\u0647", "\u0654"): "\u06C0",  # ه + U+0654 -> ۀ
    ("\u06CC", "\u0654"): "\u0626",  # ی + U+0654 -> ئ
    ("\u064A", "\u0654"): "\u0626",  # ي + U+0654 -> ئ
    ("\u0648", "\u0654"): "\u0624",  # و + U+0654 -> ؤ
    ("\u0627", "\u0654"): "\u0623",  # ا + U+0654 -> أ
    ("\u0627", "\u0655"): "\u0625",  # ا + U+0655 -> إ
}

# Harf ile hemze arasına girebilecek harekelere izin veren desen:
# Grup 1: Taban harf (ه, ی, ي, و, ا)
# Grup 2: Araya giren harekeler (korunur, sonraki hareke adımında işlenir)
# Grup 3: Birleşik hemze (U+0654 veya U+0655)
HAMZA_COMBINE_PATTERN: regex.Pattern = regex.compile(
    r"([\u0647\u06CC\u064A\u0648\u0627])([\u064B-\u0653\u0656-\u065F\u0670]*)([\u0654\u0655])"
)

# 7. Harf Birleştirme Eşlemeleri (Letter Unification)
LETTER_UNIFICATION_MAP: dict[str, str] = {
    "\u064A": "\u06CC",  # ي (Arabic Yeh) -> ی (Farsi Yeh)
    "\u0649": "\u06CC",  # ى (Alef Maksura) -> ی (Farsi Yeh)
    "\u0643": "\u06A9",  # ك (Arabic Kaf) -> ک (Keheh)
    "\u06C1": "\u0647",  # ہ (Urdu Heh Goal) -> ه (Arabic Heh)
    "\u06C2": "\u06C0",  # ۂ (Urdu Heh Goal with Hamza) -> ۀ (Heh with Yeh Above)
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
    "\u06C0",  # ۀ (Heh with Yeh Above / İzafet)
}

# 8. Hemze Katlama (Fold Hamza)
HAMZA_FOLDING_MAP: dict[str, str] = {
    "\u0671": "\u0627",  # ٱ (Alef Wasla) -> ا
    "\u0625": "\u0627",  # إ (Alef with Hamza Below) -> ا
    "\u0623": "\u0627",  # أ (Alef with Hamza Above) -> ا
}

# 9. Rakam Eşlemeleri (Arap-Hint ve Farsça -> ASCII 0-9)
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

# 10. Noktalama İşaretleri Eşlemesi
PUNCTUATION_MAP: dict[str, str] = {
    "\u060C": ",",  # ، (Arabic Comma)
    "\u061F": "?",  # ؟ (Arabic Question Mark)
    "\u061B": ";",  # ؛ (Arabic Semicolon)
}

# 11. Boşluk Deseni
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
    1. NFC normalizasyonu
    1.a. Hareke sunum formlarını (U+FE70–U+FE7F) boşluksuz birleşik harekeye çevirme
    1.b. Harf sunum formlarını (U+FB50–U+FDFF, U+FE80–U+FEFC) NFKC ile çözme
    2. Yön kontrol karakterlerini silme
    3. Sıfır genişlikli karakterleri (U+200B, U+200D) silme (U+200C / ZWNJ korunur)
    4. Tatvil (U+0640) silme
    5. Harf + birleşik hemze birleştirme (ه+U+0654->ۀ, ی/ي+U+0654->ئ, و+U+0654->ؤ, ا+U+0654->أ, ا+U+0655->إ)
    6. keep_harakat=False ise harekeleri silme (U+0654 ve U+0655 korunur)
    7. Harf birleştirme (ي, ى -> ی, ك -> ک, ہ -> ه, ۂ -> ۀ)
    8. fold_hamza=True ise ٱ, إ, أ -> ا dönüştürme
    9. ascii_digits=True ise Arap-Hint ve Farsça rakamları 0-9'a çevirme
    10. Arapça noktalama işaretlerini (, ? ;) çevirme
    11. Tüm boşlukları tek boşluğa indirip baştan/sondan kırpma
    """
    if not text:
        return ""

    # 1. NFC normalizasyonu
    text = unicodedata.normalize("NFC", text)

    # 1.a. Hareke sunum formlarını (U+FE70-U+FE7F) boşluksuz birleşik harekeye çevir
    if any(ch in HARAKAT_PRESENTATION_MAP for ch in text):
        text = "".join(HARAKAT_PRESENTATION_MAP.get(ch, ch) for ch in text)

    # 1.b. Harf sunum formlarına NFKC uygula
    text = PRESENTATION_FORMS_PATTERN.sub(
        lambda m: unicodedata.normalize("NFKC", m.group(0)), text
    )

    # 2, 3, 4. Yön kontrolleri, sıfır genişlikli karakterler ve tatvili temizle
    cleaned_chars: list[str] = []
    for ch in text:
        if ch in DIRECTIONAL_CHARS:
            continue
        if ch in ZERO_WIDTH_CHARS_TO_REMOVE:
            continue
        if ch == TATWEEL_CHAR:
            continue
        cleaned_chars.append(ch)
    text = "".join(cleaned_chars)

    # 5. Harf + birleşik hemze çiftlerini tek harfe çevir (hareke adımından önce)
    def _combine_hamza(match: regex.Match) -> str:
        base = match.group(1)
        intermediate_harakat = match.group(2)
        hamza = match.group(3)
        rep = HAMZA_BASE_MAP.get((base, hamza))
        if rep is not None:
            return rep + intermediate_harakat
        return match.group(0)

    text = HAMZA_COMBINE_PATTERN.sub(_combine_hamza, text)

    # 6, 7, 8, 9, 10. Karakter bazlı dönüşümler
    result_chars: list[str] = []
    for ch in text:
        # 6. Harekeleri sil (U+0654 ve U+0655 bu kümede yoktur, korunur)
        if not keep_harakat and ch in HARAKAT_CHARS:
            continue

        # 7. Harf birleştirme (ي, ى -> ی, ك -> ک, ہ -> ه, ۂ -> ۀ)
        if ch in LETTER_UNIFICATION_MAP:
            ch = LETTER_UNIFICATION_MAP[ch]

        # 8. fold_hamza=True ise hemzeleri katla
        if fold_hamza and ch in HAMZA_FOLDING_MAP:
            ch = HAMZA_FOLDING_MAP[ch]

        # 9. Rakam dönüştürme
        if ascii_digits and ch in DIGIT_TRANSLATION_MAP:
            ch = DIGIT_TRANSLATION_MAP[ch]

        # 10. Noktalama dönüştürme
        if ch in PUNCTUATION_MAP:
            ch = PUNCTUATION_MAP[ch]

        result_chars.append(ch)

    # 11. Boşlukları tek boşluğa indir ve kırp
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
