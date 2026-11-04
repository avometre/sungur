"""Wikitext temizleme ve poem koruma test scripti."""

import re
import sys
import mwparserfromhell

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sample_wikitext = """
{{eser başlığı|başlık=Test Eseri|yazar=Ömer Seyfettin|yıl=1919}}
{| class="wikitable"
|-
| Tablo hücresi
|}
Bu bir giriş cümlesidir. Burada ikinci bir cümle vardır.<ref>Bu bir dipnottur.</ref>
[[Dosya:Resim.jpg|küçük|Açıklama]]
<poem>
Birinci mısra burada yer alır,
İkinci mısra burada biter.
Üçüncü mısra da devam eder.
</poem>
Bu ise düzyazı kapanış paragrafıdır. Son cümle buradadır.
"""

def clean_wikitext(wikitext: str):
    # 1. <poem> etiketlerini korumak için geçici belirteçle sakla
    poem_blocks = []
    def poem_repl(match):
        idx = len(poem_blocks)
        content = match.group(1)
        poem_blocks.append(content)
        return f"\n__POEM_BLOCK_{idx}__\n"
    
    # <poem>...</poem> içeriğini ayıkla
    text_with_tokens = re.sub(r"<poem[^>]*>(.*?)</poem>", poem_repl, wikitext, flags=re.DOTALL | re.IGNORECASE)
    
    # 2. mwparserfromhell ile parse et
    parsed = mwparserfromhell.parse(text_with_tokens)
    
    # Şablonları kaldır
    for t in parsed.filter_templates():
        try:
            parsed.remove(t)
        except ValueError:
            pass
            
    # Tabloları ve ref etiketlerini kaldır
    for tag in parsed.filter_tags():
        tag_name = str(tag.tag).lower()
        if tag_name in ("ref", "table", "td", "tr", "th"):
            try:
                parsed.remove(tag)
            except ValueError:
                pass
                
    # Düz metne dönüştür
    plain = parsed.strip_code()
    
    # Dosya/Resim bağlantılarını kaldır ([[Dosya:...]] veya [[Resim:...]])
    plain = re.sub(r"\[\[(?:Dosya|Resim|File|Image):[^\]]+\]\]", "", plain, flags=re.IGNORECASE)
    
    # Bölüm başlıklarını ve fazla boşlukları temizle
    lines = []
    for line in plain.split("\n"):
        line = line.strip()
        if not line or line.startswith("=="):
            continue
        
        # Poem belirteci mi?
        m = re.match(r"^__POEM_BLOCK_(\d+)__$", line)
        if m:
            p_idx = int(m.group(1))
            poem_raw = poem_blocks[p_idx]
            # Poem içindeki mısraları satır satır temizle ve ekle
            for p_line in poem_raw.split("\n"):
                p_clean = mwparserfromhell.parse(p_line).strip_code().strip()
                if p_clean:
                    lines.append(("poem", p_clean))
        else:
            lines.append(("prose", line))
            
    return lines

result = clean_wikitext(sample_wikitext)
print("Sonuç:")
for kind, line in result:
    print(f"[{kind}] {line}")
