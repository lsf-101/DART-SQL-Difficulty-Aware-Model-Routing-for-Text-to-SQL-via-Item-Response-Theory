"""Nearest-difficulty model routing for BIRD/Spider questions.

The router consumes continuous anchor-based locations produced by ``test_2PL.py``.
It never converts them into easy/moderate/hard classes.  For each unseen question
it selects the K historical samples with smallest absolute anchor-score distance,
softly weights this subset by difficulty proximity and delexicalized question
skeleton similarity, estimates each candidate model with a Beta posterior, and
applies the paper's quality-constrained utility rule.
Retrieval ablations can instead use skeleton top-K or all historical samples.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy.stats import beta as beta_distribution
from dataset_utils import normalize_dataset, dataset_fingerprint


TOKEN_COLUMNS = ("total_tokens", "token_consumption", "token_usage", "tokens")
DEFAULT_LOCAL_MODEL_CONFIG = Path("your path")
ROUTER_METHOD_VERSION = "difficulty_knn_router_v3"
NUMERICAL_EPSILON = 1e-12


@dataclass(frozen=True)
class Evaluation:
    correct: float
    tokens: float
    token_source: str


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8-sig") as file:
        return json.load(file)


def load_embeddings(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        value = np.load(path, allow_pickle=False)
    elif path.suffix.lower() in {".pkl", ".pickle"}:
        with path.open("rb") as file:
            value = pickle.load(file)
    else:
        raise ValueError(f"Unsupported embedding file: {path}; use .npy or .pkl")
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError(f"Embeddings must be a 2-D matrix, got shape {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"Embeddings contain non-finite values: {path}")
    return array


def canonical_id(value: Any) -> str:
    text = str(value).strip()
    try:
        number = float(text)
        if number.is_integer():
            return str(int(number))
    except ValueError:
        pass
    return text


def item_id(item: dict[str, Any], fallback_index: int) -> str:
    return canonical_id(item.get("question_id", fallback_index))


def parse_float(value: Any) -> float | None:
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def require_float(
    item: dict[str, Any], key: str, *, minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    value = parse_float(item.get(key))
    if value is None:
        raise ValueError(f"Every item must contain a finite {key!r}; run test_2PL.py first")
    if minimum is not None and value < minimum:
        raise ValueError(f"{key}={value} is below the allowed minimum {minimum}")
    if maximum is not None and value > maximum:
        raise ValueError(f"{key}={value} is above the allowed maximum {maximum}")
    return value


def load_local_model_candidates(path: Path) -> tuple[set[str], dict[str, Any]]:
    raw = load_json(path)
    if not isinstance(raw, dict) or raw.get("policy") != "local_only":
        raise ValueError(f"{path} must declare policy='local_only'")
    entries = raw.get("models")
    if not isinstance(entries, list):
        raise ValueError(f"{path} must contain a models array")
    names: set[str] = set()
    disabled: list[str] = []
    for entry in entries:
        if isinstance(entry, str):
            name, runtime, enabled = entry.strip(), "local", True
        elif isinstance(entry, dict):
            name = str(entry.get("name", "")).strip()
            runtime = str(entry.get("runtime", entry.get("type", ""))).strip().lower()
            enabled = bool(entry.get("enabled", True))
        else:
            raise ValueError(f"Invalid model entry in {path}: {entry!r}")
        if not name:
            raise ValueError(f"Every model entry in {path} must have a non-empty name")
        if runtime != "local":
            raise ValueError(
                f"Model {name!r} is marked runtime={runtime!r}; local_only accepts only local"
            )
        if enabled:
            if name in names:
                raise ValueError(f"Duplicate enabled model {name!r} in {path}")
            names.add(name)
        else:
            disabled.append(name)
    if not names:
        raise ValueError(f"No enabled local candidate models found in {path}")
    return names, {
        "policy": "local_only",
        "config_path": str(path),
        "eligible_models": sorted(names),
        "disabled_models": sorted(disabled),
    }


def extract_tokens(
    row: dict[str, str], *, allow_estimated_tokens: bool = False
) -> tuple[float | None, str]:
    lowered = {key.lower(): value for key, value in row.items()}
    for key in TOKEN_COLUMNS:
        value = parse_float(lowered.get(key))
        if value is not None and value > 0:
            return value, key
    prompt = parse_float(lowered.get("prompt_tokens"))
    completion = parse_float(lowered.get("completion_tokens"))
    if prompt is not None or completion is not None:
        total = max(0.0, prompt or 0.0) + max(0.0, completion or 0.0)
        if total > 0:
            return total, "prompt+completion"
    if allow_estimated_tokens:
        sql = lowered.get("generated_sql", "")
        if sql:
            return max(1.0, len(sql.encode("utf-8")) / 4.0), "estimated_from_generated_sql"
    return None, "missing_or_zero"


def load_model_results(
    directory: Path,
    eligible_models: set[str] | None = None,
    *,
    allow_estimated_tokens: bool = False,
    dataset_name: str | None = None,
    source_dataset_fingerprint: str | None = None,
) -> tuple[dict[str, dict[str, Evaluation]], dict[str, Any]]:
    result: dict[str, dict[str, Evaluation]] = {}
    paths = sorted(directory.glob("*_results.csv"))
    if not paths:
        raise FileNotFoundError(f"No *_results.csv files found under {directory}")
    excluded: list[dict[str, str]] = []
    missing_or_zero_rows: dict[str, int] = {}
    unverified_files = []
    for path in paths:
        model_name = path.name[: -len("_results.csv")]
        if eligible_models is not None and model_name not in eligible_models:
            excluded.append(
                {"model": model_name, "file": str(path), "reason": "not_in_local_candidate_whitelist"}
            )
            continue
        rows: dict[str, Evaluation] = {}
        metadata_path = Path(str(path) + ".schema.json")
        if metadata_path.is_file():
            metadata = load_json(metadata_path)
            if dataset_name and metadata.get("dataset_name", dataset_name) != dataset_name:
                raise ValueError(f"Dataset mismatch in {metadata_path}")
            if source_dataset_fingerprint and metadata.get("source_dataset_fingerprint", source_dataset_fingerprint) != source_dataset_fingerprint:
                raise ValueError(f"Training dataset fingerprint mismatch in {metadata_path}")
        else:
            unverified_files.append(str(path))
        seen_ids = set()
        invalid_tokens = 0
        with path.open("r", encoding="utf-8-sig", newline="") as file:
            reader = csv.DictReader(file)
            required = {"question_id", "execution_result"}
            if not reader.fieldnames or not required.issubset(reader.fieldnames):
                raise ValueError(f"{path} must contain columns {sorted(required)}")
            for row in reader:
                identifier = canonical_id(row["question_id"])
                if identifier in seen_ids:
                    raise ValueError(f"Duplicate question_id {identifier} in {path}")
                seen_ids.add(identifier)
                correct = parse_float(row.get("execution_result"))
                if correct is None:
                    continue
                tokens, source = extract_tokens(row, allow_estimated_tokens=allow_estimated_tokens)
                if tokens is None:
                    invalid_tokens += 1
                    continue
                rows[canonical_id(row["question_id"])] = Evaluation(
                    correct=float(np.clip(correct, 0.0, 1.0)),
                    tokens=tokens,
                    token_source=source,
                )
        if invalid_tokens:
            missing_or_zero_rows[model_name] = invalid_tokens
            if not allow_estimated_tokens:
                raise ValueError(
                    f"Local candidate {model_name!r} has {invalid_tokens} rows with missing/zero "
                    "token usage. Regenerate measured token records, or use "
                    "--allow-estimated-tokens only for a non-paper compatibility run."
                )
        if not rows:
            raise ValueError(f"Local candidate {model_name!r} has no usable evaluation rows")
        result[model_name] = rows
    if not result:
        expected = sorted(eligible_models) if eligible_models is not None else "all models"
        raise ValueError(
            f"No usable local candidate result files under {directory}; expected one of {expected}"
        )
    found_names = {path.name[: -len("_results.csv")] for path in paths}
    return result, {
        "loaded_models": sorted(result),
        "excluded_result_files": excluded,
        "eligible_models_without_result_file": sorted((eligible_models or set()) - found_names),
        "missing_or_zero_token_rows": missing_or_zero_rows,
        "allow_estimated_tokens": bool(allow_estimated_tokens),
        "files_without_dataset_metadata": unverified_files,
    }


def normalized_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.where(norms > 0, norms, 1.0)


def reverse_minmax_utility(values: dict[str, float]) -> dict[str, float]:
    finite = {key: value for key, value in values.items() if math.isfinite(value)}
    if not finite:
        return {key: 0.0 for key in values}
    low, high = min(finite.values()), max(finite.values())
    if math.isclose(low, high):
        return {key: 1.0 for key in values}
    return {
        key: 1.0 - (value - low) / (high - low + NUMERICAL_EPSILON)
        if key in finite else 0.0
        for key, value in values.items()
    }


def continuous_position_from_score(
    historical_scores: np.ndarray, query_score: float
) -> tuple[int, float]:
    """Paper equation 3.2.3, recomputed against the routing history set."""
    scores = np.asarray(historical_scores, dtype=np.float64)
    if scores.ndim != 1 or scores.size == 0 or not np.all(np.isfinite(scores)):
        raise ValueError("historical_scores must be a non-empty finite vector")
    if not math.isfinite(query_score):
        raise ValueError("query_score must be finite")
    insertion_rank = 1 + int(np.count_nonzero(scores < query_score))
    position = float((insertion_rank - 1) / scores.size)
    return insertion_rank, position


def build_difficulty_neighbors(
    train_scores: np.ndarray, test_score: float, top_k: int
) -> np.ndarray:
    """Nearest min(K,n) by |s_i-s_x|; exact ties retain input row order."""
    if isinstance(top_k, bool) or not isinstance(top_k, (int, np.integer)) or top_k <= 0:
        raise ValueError("top_k must be a positive integer")
    scores = np.asarray(train_scores, dtype=np.float64)
    if scores.ndim != 1 or scores.size == 0 or not np.all(np.isfinite(scores)):
        raise ValueError("train_scores must be a non-empty finite vector")
    if not math.isfinite(test_score):
        raise ValueError("test_score must be finite")
    return np.argsort(np.abs(scores - test_score), kind="stable")[:top_k]


def joint_local_weights(
    test_score: float,
    train_scores: np.ndarray,
    test_skeleton: np.ndarray,
    train_skeletons: np.ndarray,
    difficulty_weight: float,
    skeleton_weight: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if difficulty_weight < 0 or skeleton_weight < 0:
        raise ValueError("difficulty_weight and skeleton_weight must be non-negative")
    difficulty_proximity = np.clip(1.0 - np.abs(train_scores - test_score), 0.0, 1.0)
    skeleton_similarity = np.clip(0.5 * (1.0 + train_skeletons @ test_skeleton), 0.0, 1.0)
    logits = difficulty_weight * difficulty_proximity + skeleton_weight * skeleton_similarity
    logits -= float(np.max(logits))
    values = np.exp(logits)
    weights = values / float(values.sum())
    return weights, difficulty_proximity, skeleton_similarity


def select_models(
    test_items: list[dict[str, Any]],
    train_items: list[dict[str, Any]],
    test_skeleton_embeddings: np.ndarray,
    train_skeleton_embeddings: np.ndarray,
    evaluations: dict[str, dict[str, Evaluation]],
    *,
    top_k: int = 100,
    retrieval_mode: str = "difficulty",
    local_weights: tuple[float, float] = (1.0, 1.0),
    utility_weights: tuple[float, float, float] = (0.6, 0.2, 0.2),
    alpha0: float = 1.0,
    beta0: float = 1.0,
    position_uncertainty_kappa: float = 1.0,
    quality_quantile: float = 0.05,
    quality_tolerance: float = 0.02,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if retrieval_mode not in {"difficulty", "skeleton", "global"}:
        raise ValueError(f"Unsupported retrieval mode: {retrieval_mode}")
    if isinstance(top_k, bool) or not isinstance(top_k, (int, np.integer)) or top_k <= 0:
        raise ValueError("top_k must be a positive integer")
    if not train_items:
        raise ValueError("Historical training set is empty")
    if len(test_items) != len(test_skeleton_embeddings):
        raise ValueError("Test JSON and test skeleton embeddings have different lengths")
    if len(train_items) != len(train_skeleton_embeddings):
        raise ValueError("Train JSON and train skeleton embeddings have different lengths")
    if train_skeleton_embeddings.shape[1] != test_skeleton_embeddings.shape[1]:
        raise ValueError("Train and test skeleton embedding dimensions differ")
    if alpha0 <= 0 or beta0 <= 0:
        raise ValueError("Beta prior parameters must be positive")
    if position_uncertainty_kappa < 0:
        raise ValueError("position_uncertainty_kappa must be non-negative")
    if not 0 < quality_quantile < 0.5:
        raise ValueError("quality_quantile must be in (0, 0.5)")
    if quality_tolerance < 0:
        raise ValueError("quality_tolerance must be non-negative")
    if any(weight < 0 for weight in utility_weights) or not math.isclose(
        sum(utility_weights), 1.0, abs_tol=1e-9
    ):
        raise ValueError("utility_weights must be non-negative and sum to 1")
    if utility_weights[0] <= utility_weights[1] + utility_weights[2]:
        raise ValueError("Paper constraint requires accuracy_weight > token_weight + certainty_weight")

    train_scores = np.asarray(
        [require_float(item, "anchor_difficulty_score", minimum=0.0, maximum=1.0) for item in train_items],
        dtype=np.float64,
    )
    sorted_train_indices = np.argsort(train_scores, kind="stable")
    train_ids = [item_id(item, index) for index, item in enumerate(train_items)]
    if len(set(train_ids)) != len(train_ids):
        raise ValueError("Duplicate historical training question_id")
    test_ids = [item_id(item, index) for index, item in enumerate(test_items)]
    if len(set(test_ids)) != len(test_ids):
        raise ValueError("Duplicate test question_id")
    train_skeleton_norm = normalized_rows(np.asarray(train_skeleton_embeddings, dtype=np.float64))
    test_skeleton_norm = normalized_rows(np.asarray(test_skeleton_embeddings, dtype=np.float64))
    if not np.all(np.isfinite(train_skeleton_norm)) or not np.all(np.isfinite(test_skeleton_norm)):
        raise ValueError("Skeleton embeddings must contain only finite values")

    output: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    rank_by_index = np.empty(len(train_items), dtype=np.int64)
    rank_by_index[sorted_train_indices] = np.arange(1, len(train_items) + 1)
    for test_index, item in enumerate(test_items):
        test_score = require_float(item, "anchor_difficulty_score", minimum=0.0, maximum=1.0)
        # r(x) and u(x) are defined relative to the exact historical set used for
        # routing. Recompute them from s_A instead of trusting a stale/precomputed
        # class or rank produced against a different training subset.
        # These coordinates are diagnostics; neighbor selection uses score distance.
        insertion_rank, position = continuous_position_from_score(train_scores, test_score)
        supplied_rank = parse_float(item.get("difficulty_rank"))
        supplied_position = parse_float(item.get("difficulty_position"))
        position_uncertainty = parse_float(item.get("difficulty_position_uncertainty"))
        position_uncertainty = 0.0 if position_uncertainty is None else position_uncertainty
        if position_uncertainty < 0:
            raise ValueError("difficulty_position_uncertainty must be non-negative")

        if retrieval_mode == "difficulty":
            neighbor_indices = build_difficulty_neighbors(train_scores, test_score, top_k)
        elif retrieval_mode == "skeleton":
            similarities = np.clip(
                0.5 * (1.0 + train_skeleton_norm @ test_skeleton_norm[test_index]), 0.0, 1.0
            )
            neighbor_indices = np.argsort(-similarities, kind="stable")[:top_k]
        else:
            neighbor_indices = np.arange(len(train_items))
        weights, rho_d, rho_q = joint_local_weights(
            test_score,
            train_scores[neighbor_indices],
            test_skeleton_norm[test_index],
            train_skeleton_norm[neighbor_indices],
            local_weights[0],
            local_weights[1],
        )
        raw_effective_n = float(1.0 / np.square(weights).sum())
        corrected_effective_n = raw_effective_n / (
            1.0 + position_uncertainty_kappa * position_uncertainty
        )

        raw_metrics: dict[str, dict[str, Any]] = {}
        unavailable_models: dict[str, Any] = {}
        neighbor_ids = [train_ids[int(index)] for index in neighbor_indices]
        for model_name, model_rows in evaluations.items():
            missing_ids = [identifier for identifier in neighbor_ids if identifier not in model_rows]
            if missing_ids:
                unavailable_models[model_name] = {
                    "reason": "incomplete_local_history",
                    "missing_neighbor_count": len(missing_ids),
                    "missing_neighbor_ids": missing_ids,
                }
                continue
            local_results = [model_rows[identifier] for identifier in neighbor_ids]
            correctness = np.asarray([entry.correct for entry in local_results], dtype=np.float64)
            tokens = np.asarray([entry.tokens for entry in local_results], dtype=np.float64)
            empirical_accuracy = float(weights @ correctness)
            mean_tokens = float(weights @ tokens)
            alpha = alpha0 + corrected_effective_n * empirical_accuracy
            beta = beta0 + corrected_effective_n * (1.0 - empirical_accuracy)
            posterior_accuracy = float(alpha / (alpha + beta))
            posterior_uncertainty = float(
                math.sqrt(alpha * beta / ((alpha + beta) ** 2 * (alpha + beta + 1.0)))
            )
            lower_bound = float(beta_distribution.ppf(quality_quantile, alpha, beta))
            raw_metrics[model_name] = {
                "empirical_accuracy": empirical_accuracy,
                "posterior_accuracy": posterior_accuracy,
                "posterior_uncertainty": posterior_uncertainty,
                "quality_lower_bound": lower_bound,
                "weighted_tokens": mean_tokens,
                "log_token_cost": float(math.log1p(mean_tokens)),
                "effective_sample_size_raw": raw_effective_n,
                "effective_sample_size_corrected": corrected_effective_n,
                "beta_alpha": alpha,
                "beta_beta": beta,
                "token_sources": sorted({entry.token_source for entry in local_results}),
            }
        if not raw_metrics:
            raise ValueError(
                f"No candidate model has complete local evidence for test item {test_index}"
            )

        token_utility = reverse_minmax_utility(
            {name: metric["log_token_cost"] for name, metric in raw_metrics.items()}
        )
        certainty_utility = reverse_minmax_utility(
            {name: metric["posterior_uncertainty"] for name, metric in raw_metrics.items()}
        )
        best_lower_bound = max(metric["quality_lower_bound"] for metric in raw_metrics.values())
        feasible_models = {
            name
            for name, metric in raw_metrics.items()
            if metric["quality_lower_bound"] >= best_lower_bound - quality_tolerance
        }
        for name, metric in raw_metrics.items():
            metric["token_utility"] = token_utility[name]
            metric["certainty_utility"] = certainty_utility[name]
            metric["quality_feasible"] = name in feasible_models
            metric["score"] = (
                utility_weights[0] * metric["posterior_accuracy"]
                + utility_weights[1] * metric["token_utility"]
                + utility_weights[2] * metric["certainty_utility"]
            )

        selected = sorted(
            feasible_models,
            key=lambda name: (
                -raw_metrics[name]["score"],
                -raw_metrics[name]["posterior_accuracy"],
                raw_metrics[name]["weighted_tokens"],
                name,
            ),
        )[0]
        enriched = dict(item)
        enriched.setdefault("question_id", test_index)
        enriched["model"] = selected
        output.append(enriched)

        local_evidence = []
        for local_index, train_index in enumerate(neighbor_indices):
            local_evidence.append(
                {
                    "question_id": train_ids[int(train_index)],
                    "difficulty_rank": int(rank_by_index[int(train_index)]),
                    "anchor_difficulty_score": float(train_scores[int(train_index)]),
                    "difficulty_distance": float(abs(train_scores[int(train_index)] - test_score)),
                    "difficulty_proximity": float(rho_d[local_index]),
                    "skeleton_similarity": float(rho_q[local_index]),
                    "weight": float(weights[local_index]),
                }
            )
        decisions.append(
            {
                "question_id": item_id(item, test_index),
                "anchor_difficulty_score": test_score,
                "difficulty_rank": insertion_rank,
                "difficulty_position": position,
                "input_difficulty_rank": supplied_rank,
                "input_difficulty_position": supplied_position,
                "difficulty_coordinate_recomputed": True,
                "difficulty_position_uncertainty": position_uncertainty,
                "top_k": None if retrieval_mode == "global" else top_k,
                "retrieval_mode": retrieval_mode,
                "candidate_count": len(neighbor_indices),
                "neighbor_min_rank": int(rank_by_index[neighbor_indices].min()),
                "neighbor_max_rank": int(rank_by_index[neighbor_indices].max()),
                # Legacy size alias only; these are no longer symmetric rank windows.
                "window_size": len(neighbor_indices),
                "selected_model": selected,
                "quality_reference_lower_bound": best_lower_bound,
                "quality_feasible_models": sorted(feasible_models),
                "local_evidence": local_evidence,
                "metrics": raw_metrics,
                "unavailable_models": unavailable_models,
            }
        )

    report = {
        "method_version": ROUTER_METHOD_VERSION,
        "difficulty_representation": "anchor_score_and_continuous_training_rank",
        "neighborhood": {
            "type": {"difficulty": "nearest_k_absolute_difficulty",
                     "skeleton": "top_k_skeleton_similarity",
                     "global": "all_training_samples"}[retrieval_mode],
            "retrieval_mode": retrieval_mode,
            "top_k": None if retrieval_mode == "global" else top_k,
            "distance_metric": {"difficulty": "absolute_anchor_difficulty_score_difference",
                                "skeleton": "descending_rescaled_cosine_similarity",
                                "global": "none"}[retrieval_mode],
            "tie_breaker": "historical_input_order",
            "maximum_candidate_count": len(train_items) if retrieval_mode == "global" else min(top_k, len(train_items)),
            "maximum_window_size": len(train_items) if retrieval_mode == "global" else min(top_k, len(train_items)),
            "difficulty_weight": local_weights[0],
            "skeleton_weight": local_weights[1],
            "skeleton_policy": {"difficulty": "soft_weight_inside_difficulty_knn_no_skeleton_top_k",
                                "skeleton": "skeleton_top_k_then_soft_weight",
                                "global": "soft_weight_over_all_training_samples"}[retrieval_mode],
        },
        "beta_prior": {"alpha0": alpha0, "beta0": beta0},
        "position_uncertainty_kappa": position_uncertainty_kappa,
        "quality_constraint": {
            "beta_quantile": quality_quantile,
            "lower_bound_tolerance": quality_tolerance,
        },
        "utility_weights": {
            "accuracy": utility_weights[0],
            "token_efficiency": utility_weights[1],
            "certainty": utility_weights[2],
        },
        "decisions": decisions,
    }
    return output, report


def prepare_routing_items(payload, dataset_name):
    if dataset_name == "spider":
        return normalize_dataset(payload, dataset_name)
    if not isinstance(payload, list) or not payload:
        raise ValueError("Routing dataset must be a nonempty array")
    if any(item.get("dataset_name", dataset_name) != dataset_name for item in payload):
        raise ValueError("Dataset mismatch; set --dataset-name to match the position files")
    return [dict(item, question_id=item.get("question_id", i)) for i, item in enumerate(payload)]


def validate_skeleton_identity(path, items, dataset_name):
    metadata_path = Path(str(path) + ".meta.json")
    if not metadata_path.is_file():
        if dataset_name == "spider":
            raise ValueError(f"Missing skeleton identity metadata: {metadata_path}; run test_2PL.py")
        return
    metadata = load_json(metadata_path)
    expected = {"dataset_name": dataset_name,
                "ordered_question_ids": [item_id(item, i) for i, item in enumerate(items)]}
    if all("db_id" in item and "question" in item for item in items):
        expected["items_fingerprint"] = dataset_fingerprint(items)
    for key, value in expected.items():
        if (dataset_name == "spider" or key in metadata) and metadata.get(key) != value:
            raise ValueError(f"Skeleton identity/order mismatch: {path}: {key}")


def training_source_fingerprint(items):
    values = {item["source_dataset_fingerprint"] for item in items if item.get("source_dataset_fingerprint")}
    if len(values) > 1:
        raise ValueError("Historical items come from different source datasets")
    return next(iter(values), None)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-name", choices=("bird", "spider"), default="bird")
    parser.add_argument("--test-json", type=Path, required=True)
    parser.add_argument("--train-json", type=Path, required=True)
    parser.add_argument("--test-skeleton-embeddings", type=Path, required=True)
    parser.add_argument("--train-skeleton-embeddings", type=Path, required=True)
    parser.add_argument("--model-results-dir", type=Path, required=True)
    parser.add_argument(
        "--local-model-config", type=Path, default=DEFAULT_LOCAL_MODEL_CONFIG,
        help="JSON whitelist of locally available candidate models",
    )
    parser.add_argument(
        "--allow-estimated-tokens", action="store_true",
        help="Compatibility only; formal paper experiments require measured token usage",
    )
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--report-json", type=Path)
    parser.add_argument("--top-k", type=int, default=100,
                        help="Local candidate count for difficulty/skeleton retrieval; uses min(K,n)")
    parser.add_argument("--retrieval-mode", choices=("difficulty", "skeleton", "global"),
                        default="difficulty", help="Candidate retrieval; global ignores --top-k")
    parser.add_argument("--difficulty-weight", type=float, default=1.0)
    parser.add_argument("--skeleton-weight", type=float, default=1.0)
    parser.add_argument("--accuracy-weight", type=float, default=0.6)
    parser.add_argument("--token-weight", type=float, default=0.2)
    parser.add_argument("--uncertainty-weight", type=float, default=0.2)
    parser.add_argument("--beta-alpha0", type=float, default=1.0)
    parser.add_argument("--beta-beta0", type=float, default=1.0)
    parser.add_argument("--position-uncertainty-kappa", type=float, default=1.0)
    parser.add_argument("--quality-quantile", type=float, default=0.05)
    parser.add_argument("--quality-tolerance", type=float, default=0.02)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    test_items = prepare_routing_items(load_json(args.test_json), args.dataset_name)
    train_items = prepare_routing_items(load_json(args.train_json), args.dataset_name)
    if not isinstance(test_items, list) or not isinstance(train_items, list):
        raise ValueError("Train and test JSON roots must both be arrays")
    eligible_models, policy_metadata = load_local_model_candidates(args.local_model_config)
    validate_skeleton_identity(args.test_skeleton_embeddings, test_items, args.dataset_name)
    validate_skeleton_identity(args.train_skeleton_embeddings, train_items, args.dataset_name)
    evaluations, result_metadata = load_model_results(
        args.model_results_dir,
        eligible_models=eligible_models,
        allow_estimated_tokens=args.allow_estimated_tokens,
        dataset_name=args.dataset_name,
        source_dataset_fingerprint=training_source_fingerprint(train_items),
    )
    output, report = select_models(
        test_items=test_items,
        train_items=train_items,
        test_skeleton_embeddings=load_embeddings(args.test_skeleton_embeddings),
        train_skeleton_embeddings=load_embeddings(args.train_skeleton_embeddings),
        evaluations=evaluations,
        top_k=args.top_k,
        retrieval_mode=args.retrieval_mode,
        local_weights=(args.difficulty_weight, args.skeleton_weight),
        utility_weights=(args.accuracy_weight, args.token_weight, args.uncertainty_weight),
        alpha0=args.beta_alpha0,
        beta0=args.beta_beta0,
        position_uncertainty_kappa=args.position_uncertainty_kappa,
        quality_quantile=args.quality_quantile,
        quality_tolerance=args.quality_tolerance,
    )
    report["candidate_model_policy"] = policy_metadata
    report["dataset_name"] = args.dataset_name
    report["model_result_loading"] = result_metadata
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w", encoding="utf-8") as file:
        json.dump(output, file, ensure_ascii=False, indent=2)
    report_path = args.report_json or args.output_json.with_name(f"{args.output_json.stem}_report.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)
    print(f"Selected models for {len(output)} questions: {args.output_json}")
    print(f"Selection diagnostics: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
