"""Dökümdeki eser başlığı ve şablon yapılarını incele."""

import bz2
import sys
import xml.etree.ElementTree as ET
import mwparserfromhell

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

DUMP_PATH = "data/raw/_cache/trwikisource-latest-pages-articles.xml.bz2"

samples = 0
with bz2.open(DUMP_PATH, "rb") as f:
    for _, elem in ET.iterparse(f, events=("end",)):
        if elem.tag.endswith("page"):
            ns = elem.find("{*}ns")
            redirect = elem.find("{*}redirect")
            if ns is not None and ns.text == "0" and redirect is None:
                title = elem.find("{*}title").text or ""
                rev = elem.find("{*}revision")
                txt = rev.find("{*}text").text if rev is not None else ""
                if txt and ("{{eser başlığı" in txt.lower() or "{{eser" in txt.lower()):
                    code = mwparserfromhell.parse(txt)
                    for t in code.filter_templates():
                        t_name = str(t.name).strip().lower()
                        if t_name in ("eser başlığı", "eser"):
                            params = {str(p.name).strip(): str(p.value).strip() for p in t.params}
                            print(f"TITLE: {title}")
                            print(f"TEMPLATE: {t_name}")
                            print(f"PARAMS: {params}")
                            print("-" * 50)
                            samples += 1
                            break
                if samples >= 10:
                    break
            elem.clear()
