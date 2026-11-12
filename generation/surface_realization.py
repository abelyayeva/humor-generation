# -*- coding: utf-8 -*-

# !pip install pymorphy3 pymorphy2-dicts-ru pandas

import re
import pandas as pd
import pymorphy3

morph = pymorphy3.MorphAnalyzer()

POS_PRIORITY = {
    "NOUN": 10, "VERB": 9, "INFN": 9, "ADJF": 7, "ADJS": 7,
    "ADVB": 6, "PRTF": 5, "PRTS": 5, "GRND": 5,
    "NUMR": 4, "NPRO": 8, "PRED": 4, "PREP": 3,
    "CONJ": 2, "PRCL": 2, "INTJ": 2,
}

def get_best_parse(lemma: str):
    """
    Возвращает наиболее вероятный разбор леммы.
    Для леммы pymorphy2 обычно даёт несколько вариантов (омонимия),
    поэтому берём максимум по (score, POS_PRIORITY).
    """
    parses = morph.parse(lemma)
    if not parses:
        return None
    def key(p):
        return (round(p.score, 3), POS_PRIORITY.get(p.tag.POS, 0))
    parses_sorted = sorted(parses, key=key, reverse=True)
    return parses_sorted[0]


def classify_lemma(lemma: str) -> str:
    """Возвращает POS-тег строкой ('NOUN', 'VERB', ...) или 'UNKN'."""
    p = get_best_parse(lemma)
    if p is None:
        return "UNKN"
    pos = str(p.tag.POS) if p.tag.POS is not None else None
    if pos is None:
        if "INFN" in p.tag:
            return "INFN"
        return "UNKN"
    return pos


def is_noun_pos(pos: str) -> bool:
    return pos in {"NOUN", "NPRO"}

def is_verb_pos(pos: str) -> bool:
    return pos in {"VERB", "INFN", "PRTF", "PRTS", "GRND"}

def is_adj_pos(pos: str) -> bool:
    return pos in {"ADJF", "ADJS", "PRTF", "PRTS"}

def is_prep_lemma(lemma: str) -> bool:
    return classify_lemma(lemma) == "PREP"

def is_conj_lemma(lemma: str) -> bool:
    return classify_lemma(lemma) == "CONJ"


CASE_TO_PYMORPHY = {
    "nom": "nomn",
    "acc": "accs",
    "dat": "datv",
    "gen": "gent",
    "ins": "ablt",
    "loc": "loct",
}

def _case_gram(case: str) -> str:
    """Возвращает граммему падежа pymorphy без мутации констант."""
    return CASE_TO_PYMORPHY.get(case, "nomn")


def inflect(lemma: str, pos_hint: str, grammemes):
    """
    Пытается просклонять лемму с учётом целевых граммем.
    grammemes — iterable строк в нотации pymorphy ('sing','nomn','masc','3per','past' ...).
    Никогда не мутирует входные данные (приводит к set локально).
    """
    grams = set(grammemes)  # локальная копия
    parses = morph.parse(lemma)
    if not parses:
        return lemma

    filtered = parses
    if pos_hint:
        filtered_pos = [p for p in parses if p.tag.POS == pos_hint]
        if filtered_pos:
            filtered = filtered_pos

    for p in filtered:
        try:
            inflected = p.inflect(grams)
        except Exception:
            inflected = None
        if inflected is not None:
            return inflected.word

    # fallback: нормальная форма лучшего разбора
    return filtered[0].normal_form


def inflect_noun(lemma: str, case: str, number: str = "sing", gender: str = None):
    """
    case: 'nom' | 'acc' | 'dat' | 'gen' | 'ins' | 'loc'
    number: 'sing' | 'plur'
    """
    grams = {number, _case_gram(case)}
    if number == "sing" and gender:
        grams.add(gender)
    return inflect(lemma, "NOUN", grams)


def inflect_adj(lemma: str, case: str, number: str = "sing", gender: str = "masc"):
    grams = {number, _case_gram(case)}
    if number == "sing":
        grams.add(gender)
    return inflect(lemma, "ADJF", grams)


def inflect_verb(lemma: str, tense: str = "pres", number: str = "sing",
                 gender: str = None, person: str = "3per"):
    if tense == "past":
        grams = {"past", number}
        if number == "sing" and gender:
            grams.add(gender)
        return inflect(lemma, "VERB", grams)
    else:
        grams = {tense, person, number}
        return inflect(lemma, "VERB", grams)


def make_infinitive(lemma: str) -> str:
    return inflect(lemma, "INFN", {"infn"})

FUNCTION_WORDS = {
    "не", "ни", "и", "а", "но", "или", "либо", "то", "же", "ли", "бы",
    "в", "во", "на", "с", "со", "из", "к", "ко", "по", "о", "об", "от",
    "до", "у", "за", "под", "над", "при", "про", "через", "для", "без",
    "это", "этот", "тот", "его", "её", "их", "свой", "весь", "каждый",
}

PREP_CASE = {
    "в":  ("acc", "loc"),
    "во": ("acc", "loc"),
    "на": ("acc", "loc"),
    "с":  ("gen", "ins"),
    "со": ("gen", "ins"),
    "из": ("gen",),
    "к":  ("dat",),
    "ко": ("dat",),
    "по": ("dat", "loc"),
    "о":  ("loc",),
    "об": ("loc",),
    "от": ("gen",),
    "до": ("gen",),
    "у":  ("gen",),
    "за": ("acc", "ins"),
    "под":("acc", "ins"),
    "над":("ins",),
    "при":("loc",),
    "про":("acc",),
    "через": ("acc",),
    "для": ("gen",),
    "без": ("gen",),
}

def pick_prep_case(prep: str, default: str = "acc") -> str:
    cases = PREP_CASE.get(prep.lower())
    if not cases:
        return default
    for c in ("acc", "loc", "gen", "dat", "ins"):
        if c in cases:
            return c
    return default


def analyze_lemmas(lemmas):
    """
    Размечает леммы по POS.
    Возвращает:
      tokens: [{'lemma', 'pos', 'index'}]
      verb_indices: список индексов глаголов (по порядку)
      nouns: [{'lemma','index','adjs','prep'}]
    """
    tokens = []
    for i, lemma in enumerate(lemmas):
        pos = classify_lemma(lemma)
        tokens.append({"lemma": lemma, "pos": pos, "index": i})

    verb_indices = [t["index"] for t in tokens if t["pos"] in {"VERB", "INFN"}]

    used_adj, used_prep = set(), set()
    nouns = []

    for i, t in enumerate(tokens):
        if t["pos"] not in {"NOUN", "NPRO"}:
            continue

        adjs = []
        prep = None

        # прилагательные в окне ±2
        for j in range(max(0, i - 2), min(len(tokens), i + 3)):
            if j == i:
                continue
            if tokens[j]["pos"] in {"ADJF", "ADJS"} and j not in used_adj:
                adjs.append(tokens[j]["lemma"])
                used_adj.add(j)

        # предлог слева в окне ±2
        for j in range(max(0, i - 2), i):
            if tokens[j]["pos"] == "PREP" and j not in used_prep:
                prep = tokens[j]["lemma"]
                used_prep.add(j)
                break

        nouns.append({"lemma": t["lemma"], "index": i, "adjs": adjs, "prep": prep})

    return {"tokens": tokens, "verb_indices": verb_indices, "nouns": nouns}


def assign_roles(analysis):
    """
    Назначает падежи именным группам. Не теряет существительные.
    Роли:
      - первое сущ. без предлога → subject (nom)
      - второе сущ. без предлога → object (acc)
      - 3-е и далее без предлога → acc (прямое дополнение) или ins (инструмент)
      - с предлогом → падеж по предлогу
    """
    nouns_sorted = sorted(analysis["nouns"], key=lambda n: n["index"])

    subj = None
    obj = None
    others = []
    free_count = 0

    for n in nouns_sorted:
        if n["prep"]:
            n["role"] = "prep"
            n["case"] = pick_prep_case(n["prep"])
        elif subj is None:
            n["role"] = "subject"
            n["case"] = "nom"
            subj = n
        elif obj is None:
            n["role"] = "object"
            n["case"] = "acc"
            obj = n
        else:
            # чередуем acc / ins, чтобы не было «датив без причины»
            n["role"] = "other"
            n["case"] = "acc" if free_count % 2 == 0 else "ins"
            free_count += 1
        others.append(n)

    analysis["subject"] = subj
    analysis["object"] = obj
    analysis["others"] = [n for n in others if n is not subj and n is not obj]
    return analysis


def determine_gender_number(lemma):
    p = get_best_parse(lemma)
    if p is None:
        return "masc", "sing"
    gender = "masc"
    if "femn" in p.tag:
        gender = "femn"
    elif "neut" in p.tag:
        gender = "neut"
    number = "plur" if "plur" in p.tag else "sing"
    return gender, number


def realize_sentence(lemmas, preserve_order=True):
    """
    lemmas: список лемм.
    preserve_order=True — выводить слова в исходном порядке лемм.
    preserve_order=False — старый SVO-порядок.
    """
    analysis = analyze_lemmas(lemmas)
    analysis = assign_roles(analysis)

    subj = analysis["subject"]
    obj = analysis["object"]
    others = analysis["others"]
    verb_indices = analysis["verb_indices"]
    tokens = analysis["tokens"]

    subj_gender, subj_number = "masc", "sing"
    if subj:
        subj_gender, subj_number = determine_gender_number(subj["lemma"])
        subj["gender"] = subj_gender
        subj["number"] = subj_number

    word_at = {}

    # 1) существительные + их прилагательные + предлог
    for n in analysis["nouns"]:
        n_gender, n_number = determine_gender_number(n["lemma"])
        n["gender"], n["number"] = n_gender, n_number

        # предлог
        if n.get("prep"):
            # предлог — отдельный токен; находим его индекс (первый подходящий PREP слева)
            prep_idx = None
            for j in range(n["index"] - 1, -1, -1):
                if tokens[j]["pos"] == "PREP" and tokens[j]["lemma"] == n["prep"]:
                    prep_idx = j
                    break
            if prep_idx is not None:
                word_at[prep_idx] = n["prep"]

        # прилагательные — на их собственных индексах
        for adj_lemma in n["adjs"]:
            # находим индекс этого прилагательного (первый ещё не занятый)
            for j in range(len(tokens)):
                if (tokens[j]["lemma"] == adj_lemma
                        and tokens[j]["pos"] in {"ADJF", "ADJS"}
                        and j not in word_at):
                    word_at[j] = inflect_adj(adj_lemma, n["case"], n_number, n_gender)
                    break

        # само существительное — на своём индексе
        word_at[n["index"]] = inflect_noun(
            n["lemma"], n["case"], n_number, n_gender
        )

    # 2) главный глагол — на своём индексе
    if verb_indices:
        v_idx = verb_indices[0]
        word_at[v_idx] = inflect_verb(
            tokens[v_idx]["lemma"], tense="pres",
            number=subj_number, gender=subj_gender, person="3per"
        )

    # 3) остальные глаголы — инфинитивы на своих индексах
    for vi in verb_indices[1:]:
        word_at[vi] = make_infinitive(tokens[vi]["lemma"])

    # 4) наречия/частицы/союзы — как есть
    for i, t in enumerate(tokens):
        if i in word_at:
            continue
        if t["pos"] in {"ADVB", "PRCL", "INTJ", "CONJ", "PREP"}:
            word_at[i] = t["lemma"]
        elif t["pos"] == "UNKN":
            word_at[i] = t["lemma"]

    if preserve_order:
        out = [word_at[i] for i in range(len(tokens)) if i in word_at]
    else:
        # старый SVO-порядок: subject (+ adjs) → verb → object/others → extras
        out = []
        if subj:
            out.append(inflect_noun(subj["lemma"], "nom", subj["number"], subj_gender))
        if verb_indices:
            out.append(word_at[verb_indices[0]])
        for n in ([obj] if obj else []) + others:
            n_gender, n_number = n["gender"], n["number"]
            if n.get("prep"):
                out.append(n["prep"])
            out.extend(inflect_adj(a, n["case"], n_number, n_gender) for a in n["adjs"])
            out.append(inflect_noun(n["lemma"], n["case"], n_number, n_gender))
        for i in range(len(tokens)):
            if i in word_at and word_at[i] not in out:
                out.append(word_at[i])

    sent = " ".join(p for p in out if p).strip()
    if sent:
        sent = sent[0].upper() + sent[1:] + "."
    return sent

def load_lemma_rows(path: str):
    """Читает CSV, где каждая строка — набор лемм, разделённых пробелом или запятой."""
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # поддержка и пробелов, и запятых
            lemmas = re.split(r"[,\s]+", line)
            lemmas = [l for l in lemmas if l]
            if lemmas:
                rows.append(lemmas)
    return rows


def process_csv(path: str, out_path: str = "realized.txt", max_rows: int = None):
    rows = load_lemma_rows(path)
    if max_rows:
        rows = rows[:max_rows]
    results = []
    for i, lemmas in enumerate(rows, 1):
        try:
            sent = realize_sentence(lemmas)
        except Exception as e:
            sent = f"<<ОШИБКА: {e}>>"
        results.append(sent)
        print(f"[{i}] {' '.join(lemmas)}")
        print(f"   -> {sent}\n")
    with open(out_path, "w", encoding="utf-8") as f:
        for s in results:
            f.write(s + "\n")
    return results

rows = load_lemma_rows("generated_sentences.csv")
for lemmas in rows[1:6]:
  print(realize_sentence(lemmas))
