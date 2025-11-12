# config/medium_aliases.py
# -*- coding: utf-8 -*-
"""
Aliases e normalização de nomes de meios de cultura.

Uso:
  from config.medium_aliases import MEDIUM_ALIASES, canonical_medium_key, resolve_medium_alias

- canonical_medium_key(s): normaliza string (upper, sem acentos, limpa pontuação)
- resolve_medium_alias(s): tenta mapear 's' para uma sigla padrão:
      1) match exato em MEDIUM_ALIASES
      2) match por prefixo (as chaves de MEDIUM_ALIASES)
      3) se começar por uma sigla conhecida (ex.: "LB", "M9") devolve essa sigla
      4) fallback: devolve a versão upper/limpa de s (para não perderes a info)
"""

import unicodedata
import re

# --- DICIONÁRIO DE ALIASES (expandido) ---
MEDIUM_ALIASES = {
    # ======================
    # Meios gerais e nutritivos
    # ======================
    "LURIA-BERTANI": "LB",
    "LB BROTH": "LB",
    "LB AGAR": "LB",
    "NUTRIENT BROTH": "NB",
    "NUTRIENT AGAR": "NA",
    "STANDARD METHODS AGAR": "SMA",
    "PLATE COUNT AGAR": "PCA",
    "R2A AGAR": "R2A",

    # ======================
    # Meios para testes clínicos e antibióticos
    # ======================
    "MUELLER-HINTON": "MH",
    "MULLER-HINTON": "MH",
    "MUELLER HINTON AGAR": "MH",
    "MUELLER HINTON BROTH": "MHB",
    "ANTIBIOTIC MEDIUM 1": "AM1",
    "ANTIBIOTIC MEDIUM 3": "AM3",

    # ======================
    # Meios de crescimento rápido / uso geral
    # ======================
    "TRYPTIC SOY BROTH": "TSB",
    "TRYPTIC SOY AGAR": "TSA",
    "SOYBEAN CASEIN DIGEST BROTH": "TSB",
    "CASO BROTH": "TSB",
    "BRAIN HEART INFUSION": "BHI",
    "BHI BROTH": "BHI",
    "BHI AGAR": "BHI",
    "SUPER BROTH": "SB",
    "TERRIFIC BROTH": "TB",
    "2XYT BROTH": "2XYT",

    # ======================
    # Meios seletivos e diferenciais
    # ======================
    "MACCONKEY": "MAC",
    "MACCONKEY AGAR": "MAC",
    "MACCONKEY BROTH": "MAC",
    "EOSIN METHYLENE BLUE AGAR": "EMB",
    "LEVINE EMB": "EMB",
    "MANNITOL SALT AGAR": "MSA",
    "XYLOSE LYSINE DEOXYCHOLATE AGAR": "XLD",
    "HEKTOEN ENTERIC AGAR": "HE",
    "SALMONELLA-SHIGELLA AGAR": "SS",
    "CLED AGAR": "CLED",
    "DEOXYCHOLATE CITRATE AGAR": "DCA",
    "BRILLIANT GREEN AGAR": "BGA",
    "BISMUTH SULFITE AGAR": "BSA",
    "VIOLET RED BILE AGAR": "VRBA",
    "VIOLET RED BILE GLUCOSE AGAR": "VRBGA",
    "BRILLIANT GREEN BILE BROTH": "BGBB",
    "M-ENDO AGAR": "ENDO",
    "M-FC AGAR": "MFC",
    "M-TEC AGAR": "MTEC",
    "M-EI AGAR": "MEI",

    # ======================
    # Meios para fungos e leveduras
    # ======================
    "SABOURAUD DEXTROSE AGAR": "SDA",
    "SABOURAUD AGAR": "SDA",
    "POTATO DEXTROSE AGAR": "PDA",
    "MALT EXTRACT AGAR": "MEA",
    "YEAST MALT AGAR": "YMA",
    "CORN MEAL AGAR": "CMA",
    "ROSE BENGAL AGAR": "RBA",
    "CHROMAGAR CANDIDA": "CHROM-C",
    "MYCOSAL AGAR": "MYC",

    # ======================
    # Meios enriquecidos / anaeróbios
    # ======================
    "CHOCOLATE AGAR": "CA",
    "BLOOD AGAR": "BA",
    "BLOOD INFUSION AGAR": "BIA",
    "COOKED MEAT MEDIUM": "RCM",
    "REINFORCED CLOSTRIDIAL MEDIUM": "RCM",
    "FASTIDIOUS ANAEROBE AGAR": "FAA",
    "THIOGLYCOLLATE BROTH": "THIO",

    # ======================
    # Micobactérias
    # ======================
    "LOWENSTEIN-JENSEN": "LJ",
    "MIDDLEBROOK 7H10": "MB7H10",
    "MIDDLEBROOK 7H11": "MB7H11",
    "DUBOS BROTH": "DUBOS",

    # ======================
    # Enterobactérias e testes bioquímicos
    # ======================
    "SIMMONS CITRATE": "CIT",
    "SIMMONS CITRATE AGAR": "CIT",
    "TRIPLE SUGAR IRON": "TSI",
    "TRIPLE SUGAR IRON AGAR": "TSI",
    "UREA AGAR": "UREA",
    "KIA AGAR": "KIA",
    "LYSINE IRON AGAR": "LIA",
    "MOTILITY INDOL UREA MEDIUM": "MIU",
    "MR-VP BROTH": "MRVP",
    "SIM MEDIUM": "SIM",
    "INDOLE BROTH": "IND",
    "PHENOL RED BROTH": "PRB",

    # ======================
    # Anaeróbios / fermentação
    # ======================
    "ROBERTSON'S COOKED MEAT MEDIUM": "RCM",
    "ANAEROBIC BROTH": "ANB",
    "PEPTONE YEAST GLUCOSE BROTH": "PYG",

    # ======================
    # Microalgas / cianobactérias
    # ======================
    "BG11": "BG11",
    "BOLD'S BASAL MEDIUM": "BBM",
    "F2P MEDIUM": "F2P",
    "WALNE MEDIUM": "WALNE",
    "F/2 MEDIUM": "F2",
    "Z8 MEDIUM": "Z8",
    "CONWAY MEDIUM": "CONWAY",
    "PROVASOLI ENRICHED SEAWATER": "PES",

    # ======================
    # Biotecnologia / eucariontes unicelulares
    # ======================
    "YEAST PEPTONE DEXTROSE": "YPD",
    "YEAST EXTRACT PEPTONE": "YEP",
    "CORN STEEP LIQUOR": "CSL",
    "M9 MINIMAL MEDIUM": "M9",
    "MOPS BUFFERED M9": "M9MOPS",
    "MINIMAL MEDIUM": "MM",
    "RICH MEDIUM": "RM",

    # ======================
    # Águas / alimentos
    # ======================
    "LAURYL SULFATE BROTH": "LSB",
    "AZIDE DEXTROSE BROTH": "ADB",
    "KF STREPTOCOCCUS AGAR": "KF",
    "MUG BROTH": "MUG",
    "EC BROTH": "ECB",
    "EC-MUG BROTH": "ECMUG",
    "BRILLIANT GREEN BILE 2% BROTH": "BGBB2",

    # ======================
    # Listeria / Staph / Pseudomonas
    # ======================
    "FRASER BROTH": "FRASER",
    "HALF FRASER BROTH": "HFRASER",
    "OXFORD AGAR": "OXFORD",
    "PALCAM AGAR": "PALCAM",
    "BAIRD-PARKER AGAR": "BP",
    "MANNITOL EGG YOLK POLYMYXIN AGAR": "MYP",
    "CETRIMIDE AGAR": "CET",
    "PSEUDOMONAS ISOLATION AGAR": "PIA",
    "PSEUDOMONAS AGAR F": "PAF",
    "PSEUDOMONAS AGAR G": "PAG",

    # ======================
    # Actinobactérias
    # ======================
    "ACTINOMYCETE AGAR": "AA",
    "STARCH CASEIN AGAR": "SCA",
    "INTERNATIONAL STREPTOMYCES PROJECT MEDIUM 2": "ISP2",
    "INTERNATIONAL STREPTOMYCES PROJECT MEDIUM 4": "ISP4",

    # ======================
    # Outros
    # ======================
    "BUFFERED PEPTONE WATER": "BPW",
    "TRYPTONE WATER": "TW",
    "ALKALINE PEPTONE WATER": "APW",
    "PEPTONE WATER": "PW",
    "SELENITE BROTH": "SEL",
    "TETRATHIONATE BROTH": "TTB",
    "LACTOSE BROTH": "LBH",
    "FERMENTATION BROTH": "FB",
}

# Siglas base reconhecidas diretamente no início do nome (para match rápido)
KNOWN_PREFIXES = {
    "LB","M9","TSB","TSA","BHI","NB","NA","YPD","YEP","SB","TB","2XYT",
    "MAC","EMB","MSA","XLD","HE","SS","CLED","DCA","BGA","BSA","VRBA","VRBGA",
    "BGBB","BGBB2","ENDO","MFC","MTEC","MEI","SDA","PDA","MEA","YMA","CMA","RBA",
    "CHROM-C","MYC","CA","BA","BIA","RCM","FAA","THIO","LJ","MB7H10","MB7H11","DUBOS",
    "CIT","TSI","UREA","KIA","LIA","MIU","MRVP","SIM","IND","PRB","ANB","PYG",
    "BG11","BBM","F2P","WALNE","F2","Z8","CONWAY","PES","MM","RM","LSB","ADB",
    "KF","MUG","ECB","ECMUG","FRASER","HFRASER","OXFORD","PALCAM","BP","MYP",
    "CET","PIA","PAF","PAG","AA","SCA","ISP2","ISP4","BPW","TW","APW","PW","SEL","TTB","LBH","FB","SMA","PCA","R2A"
}

def _strip_accents(s: str) -> str:
    return "".join(ch for ch in unicodedata.normalize("NFKD", s) if not unicodedata.combining(ch))

def canonical_medium_key(s: str) -> str:
    """Upper + sem acentos + remove (), /, -, múltiplos espaços → normaliza para matching robusto."""
    s = _strip_accents(s or "")
    s = s.upper()
    s = re.sub(r"[/()\-_,;:]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def resolve_medium_alias(s: str) -> str:
    """
    Resolve um nome livre (ex.: 'LB acidificado', 'Mueller Hinton Agar') para uma sigla:
      - match exato nos aliases,
      - match por prefixo,
      - sigla conhecida no início (LB, M9, …),
      - fallback: devolve a versão canónica (upper/limpa) para manter informação.
    """
    if not s:
        return ""
    key = canonical_medium_key(s)

    # 1) match exato
    if key in MEDIUM_ALIASES:
        return MEDIUM_ALIASES[key]

    # 2) match por prefixo
    for k, v in MEDIUM_ALIASES.items():
        if key.startswith(k):
            return v

    # 3) sigla conhecida no início
    first_token = key.split(" ")[0]
    if first_token in KNOWN_PREFIXES:
        return first_token

    # 4) fallback (não perde a info completamente)
    return key
