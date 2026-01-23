# === file: families.py ===
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Carga el fichero de abreviaturas y proporciona funciones para construir
un DataFrame de metadatos por idioma (abreviatura, nombre completo, family)
ademas de un diccionario name2family base que puede ampliarse.
"""

import os
import re
import json
import pandas as pd


NAME2FAMILY = {
    # (mismo contenido que en tu version anterior, pero exportado aqui)
    "Arabic": "Afro-Asiatic (Semitic)",
    "Kabyle": "Afro-Asiatic (Berber)",
    "Maltese": "Afro-Asiatic (Semitic)",
    "Indonesian": "Austronesian",
    "Esperanto": "Constructed",
    "Tamil": "Dravidian",
    "Armenian": "Indo-European",
    "Latgalian": "Indo-European (Baltic)",
    "Latvian": "Indo-European (Baltic)",
    "Irish": "Indo-European (Celtic)",
    "Welsh": "Indo-European (Celtic)",
    "Dutch": "Indo-European (Germanic)",
    "English": "Indo-European (Germanic)",
    "German": "Indo-European (Germanic)",
    "Swedish": "Indo-European (Germanic)",
    "Western Frisian": "Indo-European (Germanic)",
    "Bangla": "Indo-European (Indo-Aryan)",
    "Divehi": "Indo-European (Indo-Aryan)",
    "Urdu": "Indo-European (Indo-Aryan)",
    "Central Kurdish": "Indo-European (Iranian)",
    "Kurmanji Kurdish": "Indo-European (Iranian)",
    "Persian": "Indo-European (Iranian)",
    "Catalan": "Indo-European (Romance)",
    "French": "Indo-European (Romance)",
    "Galician": "Indo-European (Romance)",
    "Italian": "Indo-European (Romance)",
    "Portuguese": "Indo-European (Romance)",
    "Romanian": "Indo-European (Romance)",
    "Spanish": "Indo-European (Romance)",
    "Belarusian": "Indo-European (Slavic)",
    "Czech": "Indo-European (Slavic)",
    "Polish": "Indo-European (Slavic)",
    "Russian": "Indo-European (Slavic)",
    "Slovenian": "Indo-European (Slavic)",
    "Ukrainian": "Indo-European (Slavic)",
    "Japanese": "Japonic",
    "Georgian": "Kartvelian",
    "Thai": "Kra-Dai",
    "Basque": "Language isolate",
    "Mongolian": "Mongolic",
    "Ganda": "Niger-Congo",
    "Kinyarwanda": "Niger-Congo",
    "Swahili": "Niger-Congo",
    "Cantonese": "Sino-Tibetan",
    "Chinese (China)": "Sino-Tibetan",
    "Chinese (Hong Kong)": "Sino-Tibetan",
    "Chinese (Taiwan)": "Sino-Tibetan",
    "Hakha Chin": "Sino-Tibetan",
    "Kyrgyz": "Turkic",
    "Turkish": "Turkic",
    "Uyghur": "Turkic",
    "Uzbek": "Turkic",
    "Estonian": "Uralic (Finnic)",
    "Hungarian": "Uralic",
    "Meadow Mari": "Uralic",
    "Abkhazian": "Northwest Caucasian",
}


def robust_load_abbrev_map(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Abbreviation map file not found: {path}")
    text = open(path, 'r', encoding='utf-8').read().strip()
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return {k.strip(): v.strip() for k, v in obj.items()}
    except Exception:
        pass
    mapping = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        m = re.match(r'^([A-Za-z_]+)\s*[:\,]?\s*(.+)$', line)
        if m:
            abbr = m.group(1).strip()
            fullname = m.group(2).strip()
            mapping[abbr] = fullname
            continue
        parts = line.split()
        if len(parts) >= 2:
            mapping[parts[0].strip()] = " ".join(parts[1:]).strip()
    return mapping


def build_lang_meta(abbr_list, abbr2name, name2family):
    """Construye un DataFrame con columnas: abbr, fullname, family.
    Si falta fullname o family usa fallback y marca advertencia.
    Devuelve df_lang_meta indexada por la abreviatura.
    """
    fullname_list = []
    family_list = []
    missing_fullnames = []
    missing_families = []

    for abbr in abbr_list:
        fullname = abbr2name.get(abbr)
        if fullname is None:
            fullname = abbr
            missing_fullnames.append(abbr)
        fullname_list.append(fullname)
        family = name2family.get(fullname)
        if family is None:
            family = 'Unknown'
            missing_families.append(fullname)
        family_list.append(family)

    if missing_fullnames:
        print(f"Warning: faltan nombres completos para: {missing_fullnames}")
    if missing_families:
        print(f"Warning: faltan familias para: {sorted(set(missing_families))}")

    df_meta = pd.DataFrame({
        'abbr': abbr_list,
        'fullname': fullname_list,
        'family': family_list
    })
    df_meta = df_meta.set_index('abbr')
    return df_meta

