from __future__ import annotations

import csv
import hashlib
import json
import math
import platform
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Item:
    item_id: str
    split: str
    prompt_text: str
    relevant_ids: tuple[str, ...]


@dataclass(frozen=True)
class Query:
    query_id: str
    segment: str  # "literal" | "paraphrase"
    text: str
    graded_relevance: dict[str, int]  # item_id -> grade (0, 1, 2)


TOKEN_RE = re.compile(r"[а-яёa-z0-9]+", re.IGNORECASE)


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def build_default_dataset() -> tuple[list[Item], list[Query]]:
    """
    Самостоятельный учебный набор для задачи поиска визуально-атмосферных референсов
    (домен: подбор референсов сцен для видео/арт-продакшна).
    В train входят 12 записей (только для расчёта IDF).
    В validation входят 12 запросов (6 буквальных, 6 переформулировок/синонимов)
    и пул из 12 валидационных карточек.
    """
    items = [
        # --- TRAIN (12 элементов, только для IDF) ---
        Item("R001", "train", "неон дождь ночной город отражения асфальт", ("R001",)),
        Item("R002", "train", "туман сосновый лес рассвет холодный свет", ("R002",)),
        Item("R003", "train", "бетон лофт естественный свет тень окно", ("R003",)),
        Item("R004", "train", "пустыня закат красные дюны длинные тени", ("R004",)),
        Item("R005", "train", "подводный мир кораллы лучи света глубина", ("R005",)),
        Item("R006", "train", "винил студия тёплый свет лампа пластинки", ("R006",)),
        Item("R007", "train", "неон стекло витрина ночь мокрый тротуар", ("R007",)),
        Item("R008", "train", "туман горы ущелье пасмурно тишина", ("R008",)),
        Item("R009", "train", "бетон галерея минимализм свет стена", ("R009",)),
        Item("R010", "train", "пустыня горизонт песок жара солнце", ("R010",)),
        Item("R011", "train", "подводный мир медузы синий свет толща", ("R011",)),
        Item("R012", "train", "винил проигрыватель вечер уют плёнка", ("R012",)),
        # --- VALIDATION (12 элементов) ---
        Item("V001", "val", "неон дождь ночной город отражения мокрый асфальт", ("V001", "V002")),
        Item("V002", "val", "ночной город неон отражения дождь улица", ("V002", "V001")),
        Item("V003", "val", "туман сосновый лес рассвет прохлада", ("V003", "V004")),
        Item("V004", "val", "сосновый лес туман утренний свет", ("V004", "V003")),
        Item("V005", "val", "бетон лофт естественный свет контраст", ("V005", "V006")),
        Item("V006", "val", "бетон минимализм свет тень пространство", ("V006", "V005")),
        Item("V007", "val", "пустыня закат дюны тёплый свет", ("V007", "V008")),
        Item("V008", "val", "дюны пустыня закат песок горизонт", ("V008", "V007")),
        Item("V009", "val", "подводный мир кораллы лучи света глубина", ("V009", "V010")),
        Item("V010", "val", "подводный мир синий свет глубины", ("V010", "V009")),
        Item("V011", "val", "винил студия тёплый свет лампа", ("V011", "V012")),
        Item("V012", "val", "винил вечер уют тёплый свет", ("V012", "V011")),
    ]

    # 6 буквальных запросов (слова есть в словаре train)
    # 6 переформулировок/синонимических запросов (опорные термины отсутствуют в train)
    queries = [
        Query("Q01", "literal", "неон дождь ночной город", {"V001": 2, "V002": 2, "V007": 1}),
        Query("Q02", "paraphrase", "киберпанк ливень мегаполис", {"V001": 2, "V002": 2, "V007": 1}),
        Query("Q03", "literal", "туман сосновый лес рассвет", {"V003": 2, "V004": 2, "V008": 1}),
        Query("Q04", "paraphrase", "мгла хвойный бор заря", {"V003": 2, "V004": 2, "V008": 1}),
        Query("Q05", "literal", "бетон лофт естественный свет", {"V005": 2, "V006": 2, "V009": 1}),
        Query("Q06", "paraphrase", "железобетон индустриальный лофт дневной свет", {"V005": 2, "V006": 2, "V009": 1}),
        Query("Q07", "literal", "пустыня закат дюны", {"V007": 2, "V008": 2, "V010": 1}),
        Query("Q08", "paraphrase", "барханы пустошь сумерки", {"V007": 2, "V008": 2, "V010": 1}),
        Query("Q09", "literal", "подводный мир кораллы свет", {"V009": 2, "V010": 2, "V011": 1}),
        Query("Q10", "paraphrase", "океанское дно рифы подводная подсветка", {"V009": 2, "V010": 2, "V011": 1}),
        Query("Q11", "literal", "винил студия тёплый свет", {"V011": 2, "V012": 2, "V006": 1}),
        Query("Q12", "paraphrase", "грампластинки звукозапись ламповый свет", {"V011": 2, "V012": 2, "V006": 1}),
    ]
    return items, queries


class TfidfBaselineRetriever:
    """Простой TF-IDF + косинусное сходство без внешней словарной нормализации."""

    def __init__(self, k: int = 3, seed: int = 42) -> None:
        self.k = k
        self.seed = seed  # сохранён в конфиге; алгоритм детерминирован
        self.idf: dict[str, float] = {}
        self.doc_vectors: dict[str, dict[str, float]] = {}
        self.doc_norms: dict[str, float] = {}

    def fit_idf(self, train_items: list[Item]) -> None:
        n_docs = len(train_items)
        if n_docs == 0:
            raise ValueError("Train split is empty; cannot compute IDF.")
        df: dict[str, int] = {}
        for item in train_items:
            unique_tokens = set(tokenize(item.prompt_text))
            for tok in unique_tokens:
                df[tok] = df.get(tok, 0) + 1

        self.idf = {
            tok: math.log((1.0 + n_docs) / (1.0 + count)) + 1.0
            for tok, count in df.items()
        }

    def _vectorize(self, text: str) -> tuple[dict[str, float], float]:
        tokens = tokenize(text)
        if not tokens:
            return {}, 0.0

        tf: dict[str, int] = {}
        for tok in tokens:
            tf[tok] = tf.get(tok, 0) + 1

        weights: dict[str, float] = {}
        for tok, freq in tf.items():
            if tok in self.idf:
                weights[tok] = float(freq) * self.idf[tok]

        norm_sq = sum(w * w for w in weights.values())
        norm = math.sqrt(norm_sq)
        if norm == 0.0:
            return {}, 0.0

        normalized = {tok: w / norm for tok, w in weights.items()}
        return normalized, 1.0

    def index_candidates(self, candidate_items: list[Item]) -> None:
        self.doc_vectors.clear()
        self.doc_norms.clear()
        for item in candidate_items:
            vec, norm = self._vectorize(item.prompt_text)
            self.doc_vectors[item.item_id] = vec
            self.doc_norms[item.item_id] = norm

    def search(self, query_text: str) -> list[tuple[str, float]]:
        q_vec, q_norm = self._vectorize(query_text)
        if not q_vec or q_norm == 0.0:
            return []

        scores: list[tuple[str, float]] = []
        for doc_id, d_vec in self.doc_vectors.items():
            if not d_vec:
                continue
            # Оба вектора уже нормированы по L2 -> скалярное произведение = косинусное сходство
            sim = sum(weight * d_vec[tok] for tok, weight in q_vec.items() if tok in d_vec)
            if sim > 0.0:
                scores.append((doc_id, round(sim, 6)))

        # Детерминированная сортировка: сначала по убыванию score, при равенстве — по возрастанию item_id
        scores.sort(key=lambda pair: (-pair[1], pair[0]))
        return scores[: self.k]


def dcg_at_k(ranked_ids: list[str], grades: dict[str, int], k: int) -> float:
    score = 0.0
    for idx, doc_id in enumerate(ranked_ids[:k], start=1):
        rel = grades.get(doc_id, 0)
        gain = (2**rel) - 1
        score += gain / math.log2(idx + 1)
    return score


def ndcg_at_k(ranked_ids: list[str], grades: dict[str, int], k: int) -> float:
    actual = dcg_at_k(ranked_ids, grades, k)
    ideal_grades = sorted(grades.values(), reverse=True)[:k]
    ideal = 0.0
    for idx, rel in enumerate(ideal_grades, start=1):
        gain = (2**rel) - 1
        ideal += gain / math.log2(idx + 1)
    if ideal == 0.0:
        return 0.0
    return actual / ideal


def recall_at_k(ranked_ids: list[str], grades: dict[str, int], k: int) -> float:
    relevant_set = {doc_id for doc_id, rel in grades.items() if rel > 0}
    if not relevant_set:
        return 0.0
    hits = sum(1 for doc_id in ranked_ids[:k] if doc_id in relevant_set)
    return hits / len(relevant_set)


def config_hash(payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:12]


def run_baseline(out_dir: Path, k: int = 3, seed: int = 42) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    items, queries = build_default_dataset()

    train_items = [it for it in items if it.split == "train"]
    val_items = [it for it in items if it.split == "val"]

    retriever = TfidfBaselineRetriever(k=k, seed=seed)
    retriever.fit_idf(train_items)
    retriever.index_candidates(val_items)

    per_query_rows: list[dict[str, Any]] = []
    for q in queries:
        ranked_pairs = retriever.search(q.text)
        ranked_ids = [doc_id for doc_id, _ in ranked_pairs]
        ndcg = ndcg_at_k(ranked_ids, q.graded_relevance, k)
        rec = recall_at_k(ranked_ids, q.graded_relevance, k)
        top_grades = [q.graded_relevance.get(doc_id, 0) for doc_id in ranked_ids]

        per_query_rows.append(
            {
                "query_id": q.query_id,
                "segment": q.segment,
                "query": q.text,
                f"ndcg@{k}": round(ndcg, 4),
                f"recall@{k}": round(rec, 4),
                "top_k_ids": " ".join(ranked_ids),
                "top_k_scores": " ".join(str(sc) for _, sc in ranked_pairs),
                "top_k_grades": " ".join(str(g) for g in top_grades),
            }
        )

    def summarize(rows: list[dict[str, Any]]) -> dict[str, float]:
        if not rows:
            return {f"mean_ndcg@{k}": 0.0, f"mean_recall@{k}": 0.0}
        return {
            f"mean_ndcg@{k}": round(sum(r[f"ndcg@{k}"] for r in rows) / len(rows), 4),
            f"mean_recall@{k}": round(sum(r[f"recall@{k}"] for r in rows) / len(rows), 4),
        }

    literal_rows = [r for r in per_query_rows if r["segment"] == "literal"]
    paraphrase_rows = [r for r in per_query_rows if r["segment"] == "paraphrase"]

    summary = {
        "literal": {"count": len(literal_rows), **summarize(literal_rows)},
        "paraphrase": {"count": len(paraphrase_rows), **summarize(paraphrase_rows)},
        "all": {"count": len(per_query_rows), **summarize(per_query_rows)},
    }

    meta = {
        "dataset_version": "1.0-scene-refs",
        "k": k,
        "seed": seed,
        "deterministic": True,
        "train_count": len(train_items),
        "val_count": len(val_items),
        "query_count": len(queries),
        "python_version": platform.python_version(),
        "os": platform.platform(),
    }
    meta["config_hash"] = config_hash(meta)

    csv_path = out_dir / "validation_baseline.csv"
    fieldnames = [
        "query_id",
        "segment",
        "query",
        f"ndcg@{k}",
        f"recall@{k}",
        "top_k_ids",
        "top_k_scores",
        "top_k_grades",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(per_query_rows)

    detail_path = out_dir / "validation_baseline_detail.json"
    detail_payload = {
        "meta": meta,
        "summary": summary,
        "queries": per_query_rows,
    }
    detail_path.write_text(json.dumps(detail_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return detail_payload


if __name__ == "__main__":
    result = run_baseline(Path("results"), k=3, seed=42)
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
    print("Config hash:", result["meta"]["config_hash"])