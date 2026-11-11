import torch
import torch.nn as nn
import numpy as np
import json
import os
import pickle
import time
import hashlib
import re
import argparse
from pathlib import Path
from dataset_utils import (
    RANKER_METHOD_VERSION, SCHEMA_FILE_PATTERNS, load_dataset, normalize_dataset,
    question_key, dataset_fingerprint, render_schema_info, load_schema_map, write_json,
)
try:
    from openai import OpenAI
except ImportError:  # Required only when requesting remote embeddings.
    OpenAI = None
from embedding_input_budget import (
    DEFAULT_EMBEDDING_TOKEN_BUDGET,
    EMBEDDING_INPUT_POLICY_VERSION,
    prepare_embedding_text,
)


# ======================
# Configuration and embedding utilities shared with training for feature alignment.
# ======================
OPENAI_CONFIG = {
    "api_key": os.getenv("OPENAI_API_KEY"),
    "base_url": os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
}
openai_client = None


def get_openai_client():
    global openai_client
    if openai_client is None:
        if OpenAI is None:
            raise RuntimeError(
                "Install the optional openai dependency before generating remote embeddings"
            )
        if not OPENAI_CONFIG["api_key"]:
            raise RuntimeError("Set OPENAI_API_KEY before requesting embeddings")
        openai_client = OpenAI(
            api_key=OPENAI_CONFIG["api_key"],
            base_url=OPENAI_CONFIG["base_url"],
        )
    return openai_client

def get_batch_embeddings(texts, model, batch_size=16, max_retries=5):
    """
    Match training preprocessing by replacing all newlines with spaces before API calls.
    Retry failed requests with exponential backoff.
    """
    all_embeds = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        
        # Apply the same preprocessing as training.
        batch = [t.replace("\n", " ") for t in batch]
        # =============================================
        
        retries = 0
        wait_time = 1
        while retries < max_retries:
            try:
                response = get_openai_client().embeddings.create(
                    input=batch, model=model, timeout=30
                )
                batch_embeds = [np.array(d.embedding) for d in response.data]
                all_embeds.extend(batch_embeds)
                break
            except Exception as e:
                retries += 1
                print(f"Batch {i//batch_size+1}, attempt {retries} failed: {type(e).__name__} - {str(e)}")
                if retries >= max_retries:
                    raise RuntimeError(f"Batch failed after {max_retries} attempts") from e
                time.sleep(wait_time)
                wait_time *= 2
        time.sleep(0.5)
        if (i // batch_size + 1) % 5 == 0:
            print(f"  Embedding progress: {min(i+batch_size, len(texts))}/{len(texts)}")
    return np.array(all_embeds)

# ======================
# Schema loading and input text construction aligned with training.
# ======================
def load_tables(tables_path):
    with open(tables_path, 'r', encoding='utf-8') as f:
        tables = json.load(f)
    return {item["db_id"]: item for item in tables}

def build_embedding_text(
    item, db_schema_map, *, model="text-embedding-3-large",
    token_budget=DEFAULT_EMBEDDING_TOKEN_BUDGET, return_metadata=False,
):
    """
    Use the same text construction rules as training.
    1. Use the English labels Question, Hint, and Database Schema.
    2. Format schemas as Table `table_name` (Columns: column_1, column_2...).
    3. Preserve evidence and retain the Hint line even when empty.
    """
    db_id = item["db_id"]
    question = item["question"]
    evidence = item.get("evidence") or ""
    
    # Match the training schema string format.
    schema_full = render_schema_info(db_schema_map[db_id])
    
    # Training and test inputs share the token budget and schema-prefix truncation policy.
    try:
        prepared = prepare_embedding_text(
            question, evidence, schema_full,
            model=model, token_budget=token_budget,
        )
    except ValueError as exc:
        raise ValueError(
            f"Cannot prepare embedding input for test question_id={item.get('question_id', 'unknown')}, "
            f"db_id={db_id}: {exc}"
        ) from exc
    return (prepared.text, prepared) if return_metadata else prepared.text

# ======================
# Pairwise ranker architecture shared with training.
# ======================
class DifficultyRanker(nn.Module):
    def __init__(self, input_dim, hidden_dims, dropout):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)

def load_ranker(model_path, config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dim = config["embedding_dim"] * 4
    model = DifficultyRanker(input_dim=input_dim, hidden_dims=config["hidden_dims"], dropout=config["dropout"])
    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    except TypeError:  # Compatibility with older PyTorch releases.
        checkpoint = torch.load(model_path, map_location=device)
    if "model_state_dict" in checkpoint:
        metadata = checkpoint.get("metadata", {})
        if metadata.get("method_version") != RANKER_METHOD_VERSION:
            raise ValueError(
                "Ranker metadata does not match the full difficulty-gap multiset training method; "
                "rerun 2PL.py to regenerate the ranker"
            )
        for key in ("dataset_name", "items_fingerprint"):
            if key in config and metadata.get(key) != config[key]:
                raise ValueError(f"Ranker training input mismatch: {key}; rerun 2PL.py")
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        raise ValueError("Legacy ranker lacks method metadata; rerun 2PL.py")
    model.to(device).eval()
    print(f"Ranker loaded on device: {device}")
    return model, device

# ======================
# Select anchors near equally spaced IRT difficulty targets.
# ======================
def build_anchor_set(train_embeddings, irt_difficulties, anchor_num=20):
    train_embeddings = np.asarray(train_embeddings)
    irt_difficulties = np.asarray(irt_difficulties, dtype=np.float64)
    if len(train_embeddings) != len(irt_difficulties):
        raise ValueError("Training embedding and IRT difficulty counts differ")
    if len(irt_difficulties) == 0:
        raise ValueError("Cannot construct anchors from an empty training set")
    if not np.all(np.isfinite(irt_difficulties)):
        raise ValueError("IRT difficulties contain non-finite values")

    anchor_count = min(int(anchor_num), len(irt_difficulties))
    if anchor_count < 2:
        raise ValueError("At least two distinct anchor positions are required")

    min_diff, max_diff = irt_difficulties.min(), irt_difficulties.max()
    target_diffs = np.linspace(min_diff, max_diff, anchor_count)
    anchor_indices = []
    available = np.ones(len(irt_difficulties), dtype=bool)
    for target in target_diffs:
        distances = np.abs(irt_difficulties - target)
        distances[~available] = np.inf
        idx = int(np.argmin(distances))
        anchor_indices.append(idx)
        available[idx] = False

    anchor_indices = np.asarray(anchor_indices, dtype=np.int64)
    anchor_embeds = train_embeddings[anchor_indices]
    anchor_diffs = irt_difficulties[anchor_indices]
    print(f"Constructed {anchor_count} anchors with unique training item indices")
    print(f"Anchor IRT difficulty range: [{anchor_diffs.min():.3f}, {anchor_diffs.max():.3f}]")
    return anchor_embeds, anchor_diffs

# ======================
# Pairwise comparison
# ======================
def predict_pair_harder_prob(ranker, device, embed_a, embed_b):
    e_a = torch.FloatTensor(embed_a).to(device)
    e_b = torch.FloatTensor(embed_b).to(device)
    diff_ab = e_a - e_b
    diff_ba = e_b - e_a
    x_ab = torch.cat([e_a, e_b, diff_ab, torch.abs(diff_ab)]).unsqueeze(0)
    x_ba = torch.cat([e_b, e_a, diff_ba, torch.abs(diff_ba)]).unsqueeze(0)
    with torch.no_grad():
        p_ab = ranker(x_ab).item()
        p_ba = ranker(x_ba).item()
    # Complementary probabilities: p_phi(a>b) = (p_ab + 1 - p_ba) / 2.
    return 0.5 * (p_ab + 1.0 - p_ba)


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def embedding_input_fingerprint(texts, model):
    digest = hashlib.sha256()
    digest.update(str(model).encode("utf-8"))
    for text in texts:
        normalized = str(text).replace("\n", " ")
        encoded = normalized.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()

POSITION_METHOD_VERSION = "paper_anchor_continuous_position_v1"


def delexicalize_question(question):
    """Replace observable literal constants while preserving query intent words."""
    text = str(question)
    text = re.sub(r"(['\"]).*?\1", " <STR> ", text)
    text = re.sub(r"\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b", " <DATE> ", text)
    text = re.sub(r"\b\d+(?:\.\d+)?\s*%", " <PERCENT> ", text)
    text = re.sub(r"(?<![A-Za-z_])[-+]?\d+(?:\.\d+)?(?![A-Za-z_])", " <NUM> ", text)
    return " ".join(text.split())


def build_question_skeleton_text(item):
    return f"Question skeleton: {delexicalize_question(item['question'])}"

# 4. Compute difficulty scores for the anchors themselves.
def compute_anchor_scores(ranker, device, anchor_embeds):
    """Score each anchor with the same ranker logic used for test items."""
    anchor_scores = []
    for q, emb in enumerate(anchor_embeds):
        win_probs = []
        for ell, a_emb in enumerate(anchor_embeds):
            if q == ell:
                # Identical inputs are invariant under reversal, so self-comparison probability is 1/2.
                win_probs.append(0.5)
            else:
                win_probs.append(
                    predict_pair_harder_prob(ranker, device, emb, a_emb)
                )
        anchor_scores.append(np.mean(win_probs))
    return np.array(anchor_scores)

def predict_anchor_score(ranker, device, sample_embed, anchor_embeds):
    """Equation 3.2.2: s_A(x), not an IRT probability or a difficulty class."""
    return float(np.mean([
        predict_pair_harder_prob(ranker, device, sample_embed, anchor_embed)
        for anchor_embed in anchor_embeds
    ]))


def positions_from_historical_scores(historical_scores, query_scores):
    """Equation 3.2.3: r(x)=1+#(s_i<s_x), u(x)=(r-1)/n."""
    historical_scores = np.asarray(historical_scores, dtype=np.float64)
    query_scores = np.asarray(query_scores, dtype=np.float64)
    if historical_scores.ndim != 1 or historical_scores.size == 0:
        raise ValueError("Historical anchor scores must be a nonempty one-dimensional array")
    if not np.all(np.isfinite(historical_scores)) or not np.all(np.isfinite(query_scores)):
        raise ValueError("Anchor scores contain non-finite values")
    ranks = 1 + np.sum(
        historical_scores[None, :] < query_scores[:, None], axis=1
    )
    positions = (ranks - 1) / float(len(historical_scores))
    return ranks.astype(np.int64), positions.astype(np.float64)


def score_dataset(ranker, device, embeddings, anchor_embeds):
    scores = []
    for index, embedding in enumerate(embeddings):
        scores.append(predict_anchor_score(ranker, device, embedding, anchor_embeds))
        if (index + 1) % 100 == 0:
            print(f"  Anchor-scoring progress: {index + 1}/{len(embeddings)}")
    return np.asarray(scores, dtype=np.float64)


def load_or_create_embedding_cache(
    texts, cache_path, model, observable_input, *,
    input_policy_version=None, embedding_token_budget=None,
    truncated_question_ids=None, identity=None,
):
    cache_path = str(cache_path)
    Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
    identity = identity or {}
    metadata_path = cache_path + ".meta.json"
    fingerprint = embedding_input_fingerprint(texts, model)
    cache_valid = False
    if os.path.exists(cache_path) and os.path.exists(metadata_path):
        with open(metadata_path, "r", encoding="utf-8") as file:
            metadata = json.load(file)
        cache_valid = (
            metadata.get("fingerprint") == fingerprint
            and metadata.get("embedding_model") == model
            and metadata.get("item_count") == len(texts)
            and metadata.get("observable_input") == observable_input
            and all(metadata.get(key) == value for key, value in identity.items())
            and (
                input_policy_version is None
                or metadata.get("input_policy_version") == input_policy_version
            )
            and (
                embedding_token_budget is None
                or metadata.get("embedding_token_budget") == embedding_token_budget
            )
        )
    if cache_valid:
        with open(cache_path, "rb") as file:
            embeddings = np.asarray(pickle.load(file))
        if len(embeddings) != len(texts):
            raise ValueError(f"Embedding cache length does not match the input: {cache_path}")
        return embeddings
    embeddings = get_batch_embeddings(texts, model=model)
    with open(cache_path, "wb") as file:
        pickle.dump(embeddings, file)
    with open(metadata_path, "w", encoding="utf-8") as file:
        cache_metadata = {
                **identity,
                "fingerprint": fingerprint,
                "embedding_model": model,
                "item_count": len(texts),
                "observable_input": observable_input,
        }
        if input_policy_version is not None:
            cache_metadata["input_policy_version"] = input_policy_version
            cache_metadata["embedding_token_budget"] = embedding_token_budget
            cache_metadata["schema_truncated_question_ids"] = truncated_question_ids or []
            cache_metadata["schema_truncated_count"] = len(truncated_question_ids or [])
        json.dump(cache_metadata, file, indent=2, ensure_ascii=False)
    return np.asarray(embeddings)


def locate_unseen_difficulty(
    test_data,
    tables_path,
    ranker_path,
    train_embed_path,
    train_irt_json_path,
    config=None,
    anchor_num=20,
    output_file="your path",
    historical_output_file="your path",
    test_skeleton_cache="your path",
    train_skeleton_cache="your path",
    dataset_name="bird", schema_dir=None, schema_file_pattern="auto",
    test_embedding_cache=None,
):
    test_data = normalize_dataset(test_data, dataset_name)
    if config is None:
        config = {
            "embedding_model": "text-embedding-3-large",
            "embedding_dim": 3072,
            "embedding_token_budget": DEFAULT_EMBEDDING_TOKEN_BUDGET,
            "hidden_dims": [512, 256],
            "dropout": 0.3,
        }

    print("=" * 70)
    print(" Pairwise ranker - continuous difficulty positioning for unseen items")
    print("=" * 70)

    # Step 0. Load database schemas.
    print("\n[0/8] Loading database schemas...")
    db_schema_map = load_schema_map(test_data, schema_dir=schema_dir,
                                    tables_path=tables_path, file_pattern=schema_file_pattern)

    # Step 1. Load the ranker.
    print("\n[1/8] Loading the pairwise difficulty ranker...")
    config = dict(config)

    # Step 2. Load training embeddings and IRT difficulties.
    print("\n[2/8] Loading training data...")
    with open(train_embed_path, 'rb') as f:
        train_embeddings = pickle.load(f)
    train_metadata_path = str(train_embed_path) + ".meta.json"
    if not os.path.exists(train_metadata_path):
        raise ValueError(
            "Training embeddings lack token-budget metadata; rerun the updated 2PL.py to regenerate embeddings and the ranker"
        )
    with open(train_metadata_path, "r", encoding="utf-8") as f:
        train_embedding_metadata = json.load(f)
    token_budget = config.get("embedding_token_budget", DEFAULT_EMBEDDING_TOKEN_BUDGET)
    if (
        train_embedding_metadata.get("input_policy_version") != EMBEDDING_INPUT_POLICY_VERSION
        or train_embedding_metadata.get("embedding_token_budget") != token_budget
        or train_embedding_metadata.get("embedding_model") != config["embedding_model"]
    ):
        raise ValueError(
            "Training and test embedding models or token-budget policies differ; "
            "rerun 2PL.py with matching settings"
        )
    with open(train_irt_json_path, 'r', encoding='utf-8') as f:
        train_irt_list = normalize_dataset(json.load(f), dataset_name)
    expected_identity = {
        "dataset_name": dataset_name,
        "ordered_question_ids": [question_key(item, i) for i, item in enumerate(train_irt_list)],
        "items_fingerprint": dataset_fingerprint(train_irt_list),
    }
    for key, value in expected_identity.items():
        if (dataset_name == "spider" or key in train_embedding_metadata) and train_embedding_metadata.get(key) != value:
            raise ValueError(f"Training embedding identity or order does not match the IRT file: {key}")
    schema_format = "txt" if schema_dir is not None else "tables_json"
    if train_embedding_metadata.get("schema_source_format", schema_format) != schema_format:
        raise ValueError("Training and test schemas must use the same representation, either TXT or tables.json")
    if "dataset_name" in train_embedding_metadata:
        config.update(dataset_name=dataset_name, items_fingerprint=expected_identity["items_fingerprint"])
    ranker, device = load_ranker(ranker_path, config)
    irt_difficulties = np.array([item["irt_difficulty"] for item in train_irt_list])
    irt_discriminations = np.array([
        item.get("irt_discrimination", 1.0) for item in train_irt_list
    ], dtype=np.float64)

    if len(train_embeddings) != len(irt_difficulties):
        raise ValueError("Embedding and IRT difficulty counts differ")
    if len(irt_difficulties) != len(irt_discriminations):
        raise ValueError("IRT difficulty and discrimination counts differ")
    print(f"Loaded {len(irt_difficulties)} training items")
    print(f"2PL discrimination range: [{irt_discriminations.min():.3f}, {irt_discriminations.max():.3f}]")

    # Step 3. Construct anchors using equally spaced difficulty targets.
    print("\n[3/8] Constructing anchors across the difficulty range...")
    anchor_embeds, anchor_diffs = build_anchor_set(train_embeddings, irt_difficulties, anchor_num)

    # Step 4. Embed observable test inputs.
    print("\n[4/8] Embedding observable test inputs...")
    test_cache = str(test_embedding_cache or Path(output_file).parent / "test_embedding_cache.pkl")
    test_identity = {
        "dataset_name": dataset_name,
        "ordered_question_ids": [question_key(item, i) for i, item in enumerate(test_data)],
        "items_fingerprint": dataset_fingerprint(test_data),
    }
    prepared_test_texts = [
        build_embedding_text(
            item, db_schema_map,
            model=config["embedding_model"], token_budget=token_budget,
            return_metadata=True,
        )
        for item in test_data
    ]
    test_texts = [text for text, _ in prepared_test_texts]
    truncated_test_ids = [
        item.get("question_id", index)
        for index, (item, (_, prepared)) in enumerate(zip(test_data, prepared_test_texts))
        if prepared.truncated
    ]
    if truncated_test_ids:
        print(
            f"Truncated schemas to the {token_budget}-token embedding budget for "
            f"{len(truncated_test_ids)} test items; "
            f"example question_id values: {truncated_test_ids[:10]}"
        )
    test_embeddings = load_or_create_embedding_cache(
        test_texts, test_cache, config["embedding_model"],
        "Concat(question,evidence,schema)",
        input_policy_version=EMBEDDING_INPUT_POLICY_VERSION,
        embedding_token_budget=token_budget,
        truncated_question_ids=truncated_test_ids,
        identity=test_identity,
    )

    # Step 5. Score training and unseen items against the same fixed anchors.
    print("\n[5/8] Computing fixed-anchor scores for training and unseen items...")
    historical_scores = score_dataset(ranker, device, train_embeddings, anchor_embeds)
    test_scores = score_dataset(ranker, device, test_embeddings, anchor_embeds)
    test_ranks, test_positions = positions_from_historical_scores(
        historical_scores, test_scores
    )

    # Step 6. Embed delexicalized question skeletons for weighting retrieved candidates.
    print("\n[6/8] Generating question-skeleton embeddings...")
    train_skeleton_texts = [build_question_skeleton_text(item) for item in train_irt_list]
    test_skeleton_texts = [build_question_skeleton_text(item) for item in test_data]
    load_or_create_embedding_cache(
        train_skeleton_texts, train_skeleton_cache, config["embedding_model"],
        "delexicalized_question_skeleton",
        identity=expected_identity,
    )
    load_or_create_embedding_cache(
        test_skeleton_texts, test_skeleton_cache, config["embedding_model"],
        "delexicalized_question_skeleton",
        identity=test_identity,
    )

    # Step 7. Save shared difficulty coordinates. A single ranker records zero position variance.
    # Repeated fits or ensembles can populate the same field with sample variance across scorers.
    print("\n[7/8] Recording continuous difficulty positions...")
    historical_order = np.argsort(historical_scores, kind="stable")
    historical_ranks = np.empty(len(historical_scores), dtype=np.int64)
    historical_ranks[historical_order] = np.arange(1, len(historical_scores) + 1)
    historical_results = []
    for index, item in enumerate(train_irt_list):
        enriched = dict(item)
        enriched.update({
            "anchor_difficulty_score": float(historical_scores[index]),
            "difficulty_rank": int(historical_ranks[index]),
            "difficulty_position": float((historical_ranks[index] - 1) / len(historical_scores)),
        })
        historical_results.append(enriched)
    results = []
    for index, item in enumerate(test_data):
        enriched = dict(item)
        enriched.setdefault("question_id", index)
        enriched.update({
            "anchor_difficulty_score": float(test_scores[index]),
            "difficulty_rank": int(test_ranks[index]),
            "difficulty_position": float(test_positions[index]),
            "difficulty_position_uncertainty": 0.0,
            "difficulty_position_uncertainty_source": "single_ranker_disabled",
        })
        results.append(enriched)

    # Step 8. Save positions and the historical reference sequence.
    print("\n[8/8] Saving continuous positions...")
    write_json(output_file, results)
    write_json(historical_output_file, historical_results)
    print(f"\nUnseen-item positions saved to: {output_file}")
    print(f"Historical positions saved to: {historical_output_file}")
    return results

# ======================
# Command-line entry point
# ======================
def main(argv=None):
    parser = argparse.ArgumentParser(description="BIRD/Spider unseen difficulty positioning")
    parser.add_argument("--dataset-name", choices=("bird", "spider"), default="bird")
    parser.add_argument("--test-json", "--input-json", default="your path")
    parser.add_argument("--dataset-split", choices=("dev", "test"), default="dev")
    schemas = parser.add_mutually_exclusive_group()
    schemas.add_argument("--schema-dir")
    schemas.add_argument("--tables-json", "--tables-path")
    parser.add_argument("--schema-file-pattern", choices=("auto", *SCHEMA_FILE_PATTERNS), default="auto")
    parser.add_argument("--train-artifact-dir", required=True)
    parser.add_argument("--ranker-path")
    parser.add_argument("--train-embeddings")
    parser.add_argument("--train-irt-json")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--anchor-num", type=int, default=100)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if not args.schema_dir and not args.tables_json:
        if args.dataset_name == "spider":
            parser.error("Supply --schema-dir or --tables-json for Spider")
        args.tables_json = "your path"
    data = load_dataset(args.test_json, args.dataset_name, split=args.dataset_split)
    schemas = load_schema_map(data, args.schema_dir, args.tables_json, args.schema_file_pattern)
    root = Path(args.train_artifact_dir)
    output = Path(args.output_dir)
    if args.dry_run:
        print(json.dumps({"dataset_name": args.dataset_name, "dataset_split": args.dataset_split,
                          "question_count": len(data), "database_count": len(schemas),
                          "output_dir": str(output), "gold_sql_required": False}, indent=2))
        return 0
    ranker_path = Path(args.ranker_path or root / "irt_2pl_difficulty_ranker.pth")
    metadata_path = ranker_path.with_name(ranker_path.stem + "_metadata.json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    config = {"embedding_model": "text-embedding-3-large", "embedding_dim": 3072,
              "embedding_token_budget": DEFAULT_EMBEDDING_TOKEN_BUDGET,
              "hidden_dims": [512, 256], "dropout": 0.3}
    config.update({key: value for key, value in metadata.get("ranker_config", {}).items()
                   if key in config and value is not None})
    output.mkdir(parents=True, exist_ok=True)
    locate_unseen_difficulty(
        data, args.tables_json, str(ranker_path),
        str(args.train_embeddings or root / "2PL_embedding_cache.pkl"),
        str(args.train_irt_json or root / f"{args.dataset_name}_train_with_irt_2pl_mle.json"),
        config=config, anchor_num=args.anchor_num, dataset_name=args.dataset_name,
        schema_dir=args.schema_dir, schema_file_pattern=args.schema_file_pattern,
        output_file=str(output / f"{args.dataset_name}_{args.dataset_split}_with_anchor_position.json"),
        historical_output_file=str(output / f"{args.dataset_name}_train_with_anchor_position.json"),
        test_skeleton_cache=str(output / "test_question_skeleton_embeddings.pkl"),
        train_skeleton_cache=str(output / "train_question_skeleton_embeddings.pkl"),
        test_embedding_cache=str(output / "test_embedding_cache.pkl"),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
