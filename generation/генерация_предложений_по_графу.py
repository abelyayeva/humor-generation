# -*- coding: utf-8 -*-

import random
import re
import pandas as pd
from collections import defaultdict

FILE_PATH = "edge_final_probabilities.csv"

def parse_pair(value):
    if pd.isna(value):
        return None
    parts = re.split(r"\s*(?:→|->)\s*", str(value).strip())
    if len(parts) == 2 and all(parts):
        return parts[0], parts[1]
    return None

def load_graph(file_path):
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip()
    required = {"agent_1", "agent_2", "meaning"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Не хватает столбцов: {missing}")
    graph = defaultdict(list)
    for _, row in df.iterrows():
        a1 = str(row["agent_1"]).strip()
        a2 = str(row["agent_2"]).strip()
        meaning = str(row["meaning"]).strip()
        if pd.isna(row["agent_1"]) or pd.isna(row["agent_2"]):
            continue
        if pd.isna(row["meaning"]) or not a1 or not a2 or not meaning:
            continue
        pair = parse_pair(row.get("best_pair", None))
        if pair:
            source, target = pair
        else:
            source, target = a1, a2
        graph[source].append({
            "target": target,
            "meaning": meaning,
        })
    return dict(graph)

def generate_chain(graph, min_words=10, max_words=15, max_steps=30):
    if not graph:
        return None
    current = random.choice(list(graph.keys()))
    tokens = current.split()
    for _ in range(max_steps):
        if len(tokens) >= max_words:
            break
        edges = graph.get(current, [])
        if not edges:
            break
        edge = random.choice(edges)
        tokens.extend(edge["meaning"].split())
        tokens.extend(edge["target"].split())
        current = edge["target"]
    if min_words <= len(tokens) <= max_words:
        result = " ".join(tokens)
        return result[0].upper() + result[1:] + "."
    return None

def generate_many(
    graph,
    n=100,
    min_words=10,
    max_words=15,
    max_attempts=5000,
):
    results = set()
    for _ in range(max_attempts):
        if len(results) >= n:
            break
        sentence = generate_chain(
            graph,
            min_words=min_words,
            max_words=max_words,
        )
        if sentence is not None:
            results.add(sentence)
    return sorted(results)

if __name__ == "__main__":
    graph = load_graph(FILE_PATH)
    print("Количество начальных узлов:", len(graph))
    sentences = generate_many(
        graph,
        n=10000,
        min_words=10,
        max_words=15,
    )
    print("Уникальных цепочек:", len(sentences))
    for sentence in sentences[:50]:
        print(sentence)
    pd.DataFrame(
        {"sentence": sentences}
    ).to_csv(
        "generated_sentences.csv",
        index=False,
        encoding="utf-8-sig",
    )

