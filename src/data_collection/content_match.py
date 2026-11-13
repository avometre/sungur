"""İçerik tabanlı OCR ↔ Vikikaynak/LATOC eser eşleştirme ve satır hizalama modülü (Görev 11).

Akış:
1. printed_ocr satırları (doğrudan veya S1b çeviriyazı tahminleriyle) alınır.
2. Vikikaynak ve LATOC derlemlerindeki Latin eserlerden kelime n-gram (varsayılan 3-gram)
   ters indeksi (inverted index) oluşturulur.
3. S1b çıktılarından elde edilen n-gram'lar indekste aranarak aday eserler sıralanır.
4. Eşleşen eserlerde pencere tabanlı yerel arama ile ünsüz iskeleti benzerliği
   (skeleton_similarity >= 0.75) üzerinden satır çiftleri hizalanır.
5. Altın test/dev sızıntı kontrolünden (src/gold/leakage.py) geçirilerek
   gerçek (gürültülü OCR) cümle çiftleri yazılır.
"""

from __future__ import annotations

import argparse
import logging
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from src.common.io import read_jsonl, write_jsonl
from src.common.normalize import normalize_latin, normalize_ottoman, tr_lower
from src.gold.leakage import is_heldout, load_heldout_all
from src.translit.skeleton import skeleton_similarity

logger = logging.getLogger(__name__)

DEFAULT_PRINTED_OCR_PATH = Path("data/interim/ottoman_printed_samples.jsonl")
DEFAULT_VK_LATIN_PATH = Path("data/interim/vikikaynak_latin.jsonl")
DEFAULT_LATOC_PATH = Path("data/interim/latoc_sentences.jsonl")
DEFAULT_OUTPUT_PATH = Path("data/processed/translit_ocr_pairs.jsonl")


@dataclass
class CorpusDocument:
    """N-gram indeksine eklenecek tekil referans eser."""

    doc_id: str
    kaynak: str
    baslik: str
    satirlar: List[str]
    lisans: str = ""
    pageid: Optional[int] = None


_TR_ASCII_TABLE = str.maketrans("çğıöşü", "cgiosu")


def tokenize_for_ngram(text: str) -> List[str]:
    """Metni n-gram araması için ASCII-katlanmış küçük harf kelime listesine böler."""
    if not text:
        return []
    s = tr_lower(normalize_latin(text, strip_diacritics=True)).translate(_TR_ASCII_TABLE)
    return [w for w in re.findall(r"[a-z0-9]+", s) if len(w) >= 2]


def extract_ngrams(words: Sequence[str], n: int = 3) -> List[Tuple[str, ...]]:
    """Kelime dizisinden ardışık n-gram demetleri üretir."""
    if n <= 0 or len(words) < n:
        return []
    return [tuple(words[i : i + n]) for i in range(len(words) - n + 1)]


class CorpusNgramIndex:
    """Vikikaynak/LATOC eserleri üzerinde kelime n-gram ters indeksi."""

    def __init__(self, n: int = 3, max_df_ratio: float = 0.25) -> None:
        self.n = n
        self.max_df_ratio = max_df_ratio
        self.docs: Dict[str, CorpusDocument] = {}
        self.postings: Dict[Tuple[str, ...], set[str]] = defaultdict(set)
        self._stop_ngrams: set[Tuple[str, ...]] = set()

    def add_document(self, doc: CorpusDocument) -> None:
        self.docs[doc.doc_id] = doc
        seen: set[Tuple[str, ...]] = set()
        for line in doc.satirlar:
            words = tokenize_for_ngram(line)
            for ng in extract_ngrams(words, n=self.n):
                if ng not in seen:
                    seen.add(ng)
                    self.postings[ng].add(doc.doc_id)

    def finalize(self) -> None:
        """Çok sık geçen (ayırt edici olmayan) n-gram'ları durak listesine alır."""
        if len(self.docs) < 4:
            self._stop_ngrams.clear()
            return
        max_docs = max(2, int(len(self.docs) * self.max_df_ratio))
        self._stop_ngrams = {
            ng for ng, doc_ids in self.postings.items() if len(doc_ids) > max_docs
        }

    def search(
        self,
        query_texts: Iterable[str],
        top_k: int = 5,
        min_hits: int = 2,
    ) -> List[Tuple[CorpusDocument, int, float]]:
        """Sorgu satırlarındaki n-gram'larla en çok örtüşen aday eserleri döndürür."""
        query_ngrams: set[Tuple[str, ...]] = set()
        for text in query_texts:
            words = tokenize_for_ngram(text)
            for ng in extract_ngrams(words, n=self.n):
                if ng not in self._stop_ngrams:
                    query_ngrams.add(ng)

        if not query_ngrams:
            return []

        hit_counts: Counter[str] = Counter()
        for ng in query_ngrams:
            for doc_id in self.postings.get(ng, ()):
                hit_counts[doc_id] += 1

        results: List[Tuple[CorpusDocument, int, float]] = []
        denom = max(1, len(query_ngrams))
        for doc_id, hits in hit_counts.most_common(top_k * 2):
            if hits < min_hits:
                break
            doc = self.docs[doc_id]
            score = hits / denom
            results.append((doc, hits, score))
            if len(results) >= top_k:
                break
        return results


def _token_jaccard(text_a: str, text_b: str) -> float:
    ta = set(tokenize_for_ngram(text_a))
    tb = set(tokenize_for_ngram(text_b))
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def align_ocr_lines_to_document(
    ocr_records: Sequence[Dict[str, Any]],
    doc: CorpusDocument,
    min_skeleton: float = 0.75,
    min_arabic_chars: int = 6,
    window_radius: int = 40,
) -> List[Dict[str, Any]]:
    """OCR kitap satırlarını eşleşen referans eserin satırlarıyla hizalar.

    Her OCR satırı için önce tahmin edilen konum çevresindeki pencerede, yeterli
    aday bulunamazsa tüm belgede en yüksek (iskelet + S1b kelime örtüşmesi)
    eşleşmesini arar ve ``skeleton_similarity >= min_skeleton`` olanları seçer.
    """
    if not ocr_records or not doc.satirlar:
        return []

    ref_lines = [normalize_latin(ln).strip() for ln in doc.satirlar]
    n_ref = len(ref_lines)
    n_ocr = len(ocr_records)

    aligned: List[Dict[str, Any]] = []
    used_ref_indices: set[int] = set()
    last_ref_idx = 0

    for ocr_idx, rec in enumerate(ocr_records):
        arap = normalize_ottoman(rec.get("arap") or rec.get("metin") or "").strip()
        if len(arap.replace(" ", "")) < min_arabic_chars:
            continue

        s1b_latin = normalize_latin(rec.get("s1b_latin", "")).strip()

        # Beklenen konum ve pencere aralığı
        expected_idx = int((ocr_idx / max(1, n_ocr)) * n_ref)
        anchor = max(last_ref_idx, expected_idx) if last_ref_idx > 0 else expected_idx
        lo = max(0, anchor - window_radius)
        hi = min(n_ref, anchor + window_radius + 1)

        candidate_indices = [i for i in range(lo, hi) if i not in used_ref_indices and ref_lines[i]]
        if not candidate_indices:
            candidate_indices = [i for i in range(n_ref) if i not in used_ref_indices and ref_lines[i]]

        best_idx = -1
        best_skel = 0.0
        best_combined = -1.0

        for r_idx in candidate_indices:
            ref_lat = ref_lines[r_idx]
            skel = skeleton_similarity(arap, ref_lat)
            if skel < min_skeleton:
                continue
            lex = _token_jaccard(s1b_latin, ref_lat) if s1b_latin else 0.0
            combined = 0.75 * skel + 0.25 * lex
            if combined > best_combined:
                best_combined = combined
                best_skel = skel
                best_idx = r_idx

        if best_idx != -1:
            used_ref_indices.add(best_idx)
            last_ref_idx = best_idx
            ref_lat = ref_lines[best_idx]
            aligned.append(
                {
                    "kaynak": "printed_ocr_match",
                    "eslesen_kaynak": doc.kaynak,
                    "kitap": rec.get("kitap", ""),
                    "sayfa": rec.get("sayfa", ""),
                    "eslesen_eser": doc.baslik,
                    "eslesen_doc_id": doc.doc_id,
                    "arap": arap,
                    "latin": ref_lat,
                    "latin_sade": normalize_latin(ref_lat, strip_diacritics=True),
                    "s1b_latin": s1b_latin,
                    "iskelet_benzerlik": round(best_skel, 4),
                    "duzey": "cumle",
                    "gurultulu_ocr": True,
                    "lisans": "cc-by-nc-4.0",
                }
            )

    return aligned


def load_corpus_documents(
    vk_path: Path = DEFAULT_VK_LATIN_PATH,
    latoc_path: Path = DEFAULT_LATOC_PATH,
) -> List[CorpusDocument]:
    """Vikikaynak ve LATOC JSONL dosyalarından belge listesi yükler."""
    docs: List[CorpusDocument] = []

    if vk_path.exists():
        vk_groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for rec in read_jsonl(vk_path):
            key = str(rec.get("pageid") or rec.get("baslik") or "")
            if key:
                vk_groups[key].append(rec)

        for key, recs in vk_groups.items():
            title = recs[0].get("baslik", key)
            pageid = recs[0].get("pageid")
            lines = [r.get("latin") or r.get("metin") or "" for r in recs]
            lines = [ln.strip() for ln in lines if ln.strip()]
            if lines:
                docs.append(
                    CorpusDocument(
                        doc_id=f"vk:{key}",
                        kaynak="vikikaynak",
                        baslik=title,
                        satirlar=lines,
                        lisans="CC BY-SA 4.0",
                        pageid=int(pageid) if isinstance(pageid, int) else None,
                    )
                )

    if latoc_path.exists():
        latoc_groups: Dict[str, List[str]] = defaultdict(list)
        for rec in read_jsonl(latoc_path):
            book = str(rec.get("kitap") or rec.get("book") or rec.get("baslik") or "")
            line = (rec.get("latin") or rec.get("metin") or "").strip()
            if book and line:
                latoc_groups[book].append(line)

        for book, lines in latoc_groups.items():
            docs.append(
                CorpusDocument(
                    doc_id=f"latoc:{book}",
                    kaynak="latoc",
                    baslik=book,
                    satirlar=lines,
                    lisans="local-research-only",
                )
            )

    return docs


def match_printed_ocr_corpus(
    printed_records: Sequence[Dict[str, Any]],
    corpus_docs: Sequence[CorpusDocument],
    transliterate_fn: Optional[Callable[[List[str]], List[str]]] = None,
    ngram_size: int = 3,
    min_ngram_hits: int = 3,
    min_skeleton: float = 0.75,
    output_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """printed_ocr kayıtlarını içerik tabanlı olarak referans eserlerle eşleştirir."""
    heldout_pages, heldout_arap, heldout_titles, heldout_latin = load_heldout_all()

    index = CorpusNgramIndex(n=ngram_size)
    for doc in corpus_docs:
        index.add_document(doc)
    index.finalize()

    books: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for rec in printed_records:
        book_id = str(rec.get("kitap") or "unknown")
        books[book_id].append(dict(rec))

    all_pairs: List[Dict[str, Any]] = []
    matched_books: List[Dict[str, Any]] = []
    leaked_count = 0

    for book_id, recs in books.items():
        # Eksik s1b_latin varsa ve çevirici sağlandıysa toplu üret
        missing_idx = [i for i, r in enumerate(recs) if not r.get("s1b_latin")]
        if missing_idx and transliterate_fn is not None:
            texts = [recs[i].get("arap") or recs[i].get("metin") or "" for i in missing_idx]
            preds = transliterate_fn(texts)
            for i, pred in zip(missing_idx, preds):
                recs[i]["s1b_latin"] = pred

        queries = [r.get("s1b_latin", "") for r in recs if r.get("s1b_latin")]
        candidates = index.search(queries, top_k=3, min_hits=min_ngram_hits)
        if not candidates:
            continue

        best_doc, hits, score = candidates[0]
        aligned = align_ocr_lines_to_document(
            ocr_records=recs,
            doc=best_doc,
            min_skeleton=min_skeleton,
        )

        kept_book_pairs: List[Dict[str, Any]] = []
        for pair in aligned:
            if is_heldout(
                arap=pair["arap"],
                latin=pair["latin"],
                pageid=best_doc.pageid,
                tr_title=best_doc.baslik,
                heldout_pages=heldout_pages,
                heldout_arap=heldout_arap,
                heldout_titles=heldout_titles,
                heldout_latin=heldout_latin,
            ):
                leaked_count += 1
                continue
            kept_book_pairs.append(pair)

        if kept_book_pairs:
            all_pairs.extend(kept_book_pairs)
            matched_books.append(
                {
                    "kitap": book_id,
                    "eslesen_doc_id": best_doc.doc_id,
                    "eslesen_eser": best_doc.baslik,
                    "ngram_hits": hits,
                    "ngram_score": round(score, 4),
                    "hizalanan_satir": len(kept_book_pairs),
                }
            )

    if output_path is not None:
        write_jsonl(output_path, all_pairs)

    return {
        "toplam_kitap": len(books),
        "eslesen_kitap": len(matched_books),
        "toplam_cift": len(all_pairs),
        "elendi_sizinti": leaked_count,
        "eslesmeler": matched_books,
        "ciftler": all_pairs,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="İçerik tabanlı printed_ocr eşleştirme (Görev 11)")
    parser.add_argument("--printed", type=Path, default=DEFAULT_PRINTED_OCR_PATH)
    parser.add_argument("--vk", type=Path, default=DEFAULT_VK_LATIN_PATH)
    parser.add_argument("--latoc", type=Path, default=DEFAULT_LATOC_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--model", type=Path, default=None, help="Opsiyonel S1b modeli (s1b_latin yoksa)")
    parser.add_argument("--ngram-size", type=int, default=3)
    parser.add_argument("--min-hits", type=int, default=3)
    parser.add_argument("--min-skeleton", type=float, default=0.75)
    args = parser.parse_args()

    translit_fn = None
    if args.model is not None:
        from src.translit.infer import SungurTransliterator

        engine = SungurTransliterator(model_dir=args.model)
        translit_fn = lambda texts: engine.transliterate_batch(texts, prefix="<vk> ")

    printed_records = list(read_jsonl(args.printed)) if args.printed.exists() else []
    corpus_docs = load_corpus_documents(vk_path=args.vk, latoc_path=args.latoc)

    summary = match_printed_ocr_corpus(
        printed_records=printed_records,
        corpus_docs=corpus_docs,
        transliterate_fn=translit_fn,
        ngram_size=args.ngram_size,
        min_ngram_hits=args.min_hits,
        min_skeleton=args.min_skeleton,
        output_path=args.output,
    )
    print(
        f"Tamamlandı: {summary['eslesen_kitap']}/{summary['toplam_kitap']} kitap eşleşti, "
        f"{summary['toplam_cift']} çift yazıldı -> {args.output}"
    )


if __name__ == "__main__":
    main()
