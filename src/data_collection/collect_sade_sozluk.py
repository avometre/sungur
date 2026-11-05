"""TurkOpenDataOrg/Ottoman2Turkish_dictionary verisini ayrıştırır.

15.8 bin Latin Osmanlıca kelime -> bugünkü Türkçe anlam.
Kaynağı belirsiz olduğu için "yalnız yerel" olarak işaretlenir.
Çıktı: data/interim/sade_sozluk.jsonl
"""

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List
from huggingface_hub import hf_hub_download

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

OUTPUT_PATH = Path("data/interim/sade_sozluk.jsonl")
WORD_PATTERN = re.compile(r"['\"](.+?)['\"]\s*kelimesinin anlamı nedir\?", re.IGNORECASE)
ANSWER_PATTERN = re.compile(r"['\"].+?['\"]\s*kelimesinin anlamı:\s*(.+)$", re.IGNORECASE)


def parse_sade_sozluk(output_path: Path = OUTPUT_PATH) -> Dict[str, Any]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("TurkOpenDataOrg/Ottoman2Turkish_dictionary indiriliyor/yükleniyor...")

    file_path = hf_hub_download(
        repo_id="TurkOpenDataOrg/Ottoman2Turkish_dictionary",
        filename="data_osmanlica_fine_tuning.jsonl",
        repo_type="dataset",
    )

    records: List[Dict[str, Any]] = []
    seen: set = set()
    total_raw = 0

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            total_raw += 1
            try:
                data = json.loads(line)
                messages = data.get("messages", [])
                user_msg = next((m["content"] for m in messages if m["role"] == "user"), "")
                asst_msg = next((m["content"] for m in messages if m["role"] == "assistant"), "")

                w_match = WORD_PATTERN.search(user_msg)
                word = w_match.group(1).strip() if w_match else ""

                ans_match = ANSWER_PATTERN.search(asst_msg)
                meaning = ans_match.group(1).strip() if ans_match else asst_msg.strip()

                if word and meaning:
                    key = (word.lower(), meaning.lower())
                    if key not in seen:
                        seen.add(key)
                        records.append({
                            "kelime_latin": word,
                            "anlam_tr": meaning,
                            "kaynak": "TurkOpenDataOrg/Ottoman2Turkish_dictionary",
                            "lisans": "yalnız yerel (kaynağı belirsiz, HF kartı apache-2.0)",
                        })
            except Exception as e:
                logger.warning(f"Ayrıştırma hatası: {e}")

    with open(output_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    logger.info(f"Sade sözlük tamamlandı: {len(records)} kayıt -> {output_path}")
    stats = {
        "kaynak": "TurkOpenDataOrg/Ottoman2Turkish_dictionary",
        "lisans": "yalnız yerel (kaynağı belirsiz)",
        "ham_kayit": total_raw,
        "gecerli_kayit": len(records),
        "dosya": str(output_path),
    }
    logger.info(f"İstatistikler: {stats}")
    return stats


if __name__ == "__main__":
    parse_sade_sozluk()
