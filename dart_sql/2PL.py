import os
import argparse
from pathlib import Path
import json
import sqlite3
import pandas as pd
import numpy as np
import pickle
import threading
import time
import random
import re
import warnings
import hashlib
from scipy.optimize import OptimizeWarning
try:
    from vllm import LLM, SamplingParams
except ImportError:
    LLM = SamplingParams = None
from collections import defaultdict
from multiprocessing import Pool, cpu_count
import signal
import requests
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None
from embedding_input_budget import (
    DEFAULT_EMBEDDING_TOKEN_BUDGET, EMBEDDING_INPUT_POLICY_VERSION, prepare_embedding_text,
)
from dataset_utils import (
    RANKER_METHOD_VERSION, SCHEMA_FILE_PATTERNS, canonical_id, question_key,
    get_gold_sql, load_dataset, dataset_fingerprint, database_signature,
    read_schema_text as read_dataset_schema, load_schema_map, write_json, resolve_database_path,
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

# ==========================================
# 1. Basic configuration and paths
# ==========================================
BIRD_TRAIN_JSON = "your path"
DB_ROOT_DIR = "your path"  # SQLite database root for execution evaluation.
# Shared schema source precedence: --schema-dir, SCHEMA_DIR, then the placeholder below.
SCHEMA_DIR = os.environ.get("SCHEMA_DIR", "your path")
_SCHEMA_TEXTS = {}
OUTPUT_DIR = "your path"
FINAL_IRT_JSON = "your path"
RANKER_MODEL_PATH = "your path"
GOLD_CACHE_FILE = "your path"
EMBEDDING_CACHE_FILE = "your path"
DATASET_NAME = "bird"
TABLES_JSON = None
SCHEMA_FILE_PATTERN = "auto"
ARTIFACT_DIR = "your path"
MAX_ROW_LIMIT = 500
SQL_TIMEOUT_SEC = 3.0
DEFAULT_MAX_ROW_LIMIT = MAX_ROW_LIMIT
DEFAULT_SQL_TIMEOUT_SEC = SQL_TIMEOUT_SEC

# Local model configuration
MODELS_TO_RUN = {}

GPU_NUMS = 4
API_TIMEOUT = 30
API_BATCH_SIZE = 16
TOKEN_METRIC_COLUMNS = {"prompt_tokens", "completion_tokens", "total_tokens"}

def result_csv_has_token_metrics(csv_path, schema_fingerprint=None, expected_ids=None):
    # Retain the legacy parameter for callers; CSV reuse no longer checks fingerprints.
    if not os.path.exists(csv_path):
        return False
    try:
        header = set(pd.read_csv(csv_path, nrows=0).columns)
        if not TOKEN_METRIC_COLUMNS.issubset(header):
            return False
        usage = pd.read_csv(csv_path, dtype={"question_id": str})
        identifiers = usage["question_id"].map(canonical_id)
        if identifiers.duplicated().any():
            return False
        if expected_ids is not None and set(identifiers) != set(expected_ids):
            return False
        if not pd.to_numeric(usage["execution_result"], errors="raise").isin([0, 1]).all():
            return False
        return (
            len(usage) > 0
            and not usage[sorted(TOKEN_METRIC_COLUMNS)].isnull().any().any()
            and (usage[sorted(TOKEN_METRIC_COLUMNS)] >= 0).all().all()
            and (usage["total_tokens"] > 0).all()
            and (
                usage["prompt_tokens"] + usage["completion_tokens"]
                == usage["total_tokens"]
            ).all()
        )
    except Exception:
        return False

# ======================
# Hyperparameters
# ======================
IRT_CONFIG = {
    "vae_virtual_students": 200,    # S=200
    "sample_virtual_students": 200, # T=200
    "irt_model": "2PL",             # Two-parameter logistic IRT model.
    "estimation_method": "regularized_MLE",
    "prior": "regularized",         # Prevent discrimination parameters from saturating at the upper bound.
    "2pl_epochs": 2000,
    "2pl_learning_rate": 0.03,
    "2pl_min_discrimination": 0.05,
    "2pl_max_discrimination": 3.0,
    "2pl_real_group_weight_eta": 0.70,
    "2pl_ability_standardization_epsilon": 1e-6,
    "2pl_theta_regularization": 0.01,
    "2pl_difficulty_regularization": 0.001,
    "2pl_log_discrimination_regularization": 0.05,
}

RANKER_CONFIG = {
    "embedding_model": "text-embedding-3-large", # Embedding model.
    "embedding_dim": 3072,                       # Embedding output dimension.
    "embedding_token_budget": DEFAULT_EMBEDDING_TOKEN_BUDGET,
    "hidden_dims": [512, 256],                   # Hidden layer widths.
    "dropout": 0.3,                              # Dropout probability.
    "batch_size": 1024,                            # batch size
    "epochs": 8,                                 # Maximum epochs per candidate.
    "learning_rate": 1e-4,
    # Batch embeddings with bounded retries. Disable SDK retries to manage attempts here.
    "embedding_batch_size": 100,
    "embedding_request_timeout": 60.0,
    "embedding_max_attempts": 6,
    "embedding_retry_initial_delay": 5.0,
    "embedding_retry_max_delay": 60.0,
    "embedding_retry_jitter": 1.0,
    # Select a positive nonzero-gap quantile threshold using fixed validation pairs.
    "pair_tolerance_mode": "quantile_validation",
    "pair_tolerance_quantiles": [0.01],
    "pair_tolerance_min_retention": 0.60,
    "tolerance_selection_seed": 42,
    "tolerance_monotonic_weight": 0.10,
    "tolerance_stability_weight": 0.10,
    "tolerance_anchor_count": 32,
    "tolerance_selection_repeats": 1,
    # Used only when pair_tolerance_mode="fixed".
    "pair_tolerance": 0.0,
    "activation": "ReLU"
}
OPENAI_PROXY_CONFIG = {
    "api_key": os.getenv("OPENAI_API_KEY"),
    "base_url": os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
}
openai_client = None


def get_openai_client():
    global openai_client
    if openai_client is None:
        if OpenAI is None or not OPENAI_PROXY_CONFIG["api_key"]:
            raise RuntimeError("Install openai and configure OPENAI_API_KEY before requesting embeddings")
        openai_client = OpenAI(api_key=OPENAI_PROXY_CONFIG["api_key"],
                               base_url=OPENAI_PROXY_CONFIG["base_url"], max_retries=0)
    return openai_client

# ==========================================
# 2. Schema text loading and database evaluation utilities
# ==========================================
def read_schema_text(schema_dir, db_id):
    return read_dataset_schema(schema_dir, db_id, SCHEMA_FILE_PATTERN)


def get_db_schema_and_path(db_id):
    # SQLite paths are used only for evaluation. Prompt construction does not require database files.
    if db_id not in _SCHEMA_TEXTS:
        _SCHEMA_TEXTS.update(load_schema_map(
            [{"db_id": db_id}], schema_dir=None if TABLES_JSON else SCHEMA_DIR,
            tables_path=TABLES_JSON, file_pattern=SCHEMA_FILE_PATTERN))
    db_path = str(resolve_database_path(DB_ROOT_DIR, db_id))
    return _SCHEMA_TEXTS[db_id], db_path


def schema_input_fingerprint(bird_data):
    schemas = {db_id: get_db_schema_and_path(db_id)[0]
               for db_id in sorted({item["db_id"] for item in bird_data})}
    return hashlib.sha256(json.dumps(
        {"schemas": schemas, "dataset": dataset_fingerprint(bird_data),
         "databases": database_signature(bird_data, DB_ROOT_DIR),
         "max_rows": MAX_ROW_LIMIT, "timeout": SQL_TIMEOUT_SEC},
        ensure_ascii=False, sort_keys=True
    ).encode("utf-8")).hexdigest()


def prepare_schema_inputs(bird_data):
    """Validate all TXT files before model loading and refresh this run's snapshot."""
    _SCHEMA_TEXTS.clear()
    return schema_input_fingerprint(bird_data)

def _interrupt_handler(conn):
    try:
        conn.interrupt()
    except Exception:
        pass

# ==========================================
# 3. Reference SQL execution cache
# ==========================================
def _worker_execute_gold(args):
    db_id, items, db_root = args
    db_path = str(resolve_database_path(db_root, db_id))
    local_cache = {}
    
    if not os.path.exists(db_path):
        return db_id, local_cache
        
    conn = sqlite3.connect(Path(db_path).resolve().as_uri() + "?mode=ro", uri=True, timeout=10.0)
    cursor = conn.cursor()
    cursor.execute("PRAGMA query_only = ON;")
    
    for q_id, gold_sql in items:
        timer = threading.Timer(SQL_TIMEOUT_SEC, _interrupt_handler, args=[conn])
        timer.start()
        try:
            cursor.execute(gold_sql)
            res = cursor.fetchmany(MAX_ROW_LIMIT)
            local_cache[q_id] = pickle.dumps(set(res))
        except Exception:
            local_cache[q_id] = pickle.dumps(None)
        finally:
            timer.cancel()
            
    conn.close()
    return db_id, local_cache

def precompute_all_gold_results(bird_data):
    if DATASET_NAME == "spider2_sqlite":
        from spider2_sqlite_data import validate_reference
        combined_cache = {}
        for index, item in enumerate(bird_data):
            reference = item["spider2_reference"]
            validate_reference(reference)
            combined_cache[question_key(item, index)] = pickle.dumps(reference)
        with open(GOLD_CACHE_FILE, 'wb') as handle:
            pickle.dump(combined_cache, handle, protocol=pickle.HIGHEST_PROTOCOL)
        write_json(GOLD_CACHE_FILE + ".meta.json", {
            "fingerprint": schema_input_fingerprint(bird_data), "dataset_name": DATASET_NAME})
        print(f"Cached {len(combined_cache)} official Spider2 results; reference SQL is not required for every item.")
        return combined_cache
    print("\n>>> Executing all reference SQL queries once to populate the cache...")
    
    db_group = defaultdict(list)
    for idx, item in enumerate(bird_data):
        gold_sql = get_gold_sql(item)
        db_group[item["db_id"]].append((question_key(item, idx), gold_sql))
    
    total_dbs = len(db_group)
    print(f"Found {total_dbs} databases and {len(bird_data)} SQL queries.")
    proc_num = min(cpu_count(), 16)
    pool = Pool(processes=proc_num)
    task_args = [(dbid, items, DB_ROOT_DIR) for dbid, items in db_group.items()]
    
    combined_cache = {}
    completed_count = 0
    
    for db_id, res_dict in pool.imap_unordered(_worker_execute_gold, task_args):
        combined_cache.update(res_dict)
        completed_count += 1
        percent = (completed_count / total_dbs) * 100
        print(f"[{completed_count:3d}/{total_dbs:3d}] ({percent:6.2f}%) | Database `{db_id:<20}` reference results cached...", flush=True)
    pool.close()
    pool.join()
    print(f">>> Reference SQL cache complete: {len(combined_cache)} valid results.\n")
    try:
        with open(GOLD_CACHE_FILE, 'wb') as f_cache:
            pickle.dump(combined_cache, f_cache, protocol=pickle.HIGHEST_PROTOCOL)
        write_json(GOLD_CACHE_FILE + ".meta.json", {
            "fingerprint": schema_input_fingerprint(bird_data), "dataset_name": DATASET_NAME})
        print(f"Cache saved to: {GOLD_CACHE_FILE}\n")
    except Exception as e:
        print(f"Warning: failed to save cache: {str(e)}")
    
    return combined_cache

# ==========================================
# 4. Sandboxed SQL evaluation
# ==========================================
def single_model_db_execute(args):
    db_id, item_list, db_root, gold_cache_chunk = args
    db_path = str(resolve_database_path(db_root, db_id))
    res_list = []
    
    if not os.path.exists(db_path):
        for idx, q_id, pred_sql in item_list:
            res_list.append({"question_id": str(q_id), "generated_sql": pred_sql, "execution_result": 0})
        return db_id, res_list
        
    conn = sqlite3.connect(Path(db_path).resolve().as_uri() + "?mode=ro", uri=True, timeout=10.0)
    cursor = conn.cursor()
    cursor.execute("PRAGMA query_only = ON;")
    
    for idx, q_id in enumerate([x[1] for x in item_list]):
        pred_sql = item_list[idx][2]
        gold_bytes = gold_cache_chunk.get(q_id, None)
        if gold_bytes is None:
            is_correct = 0
        else:
            gold_set = pickle.loads(gold_bytes)
            if gold_set is None:
                is_correct = 0
            else:
                timer = threading.Timer(SQL_TIMEOUT_SEC, _interrupt_handler, args=[conn])
                timer.start()
                try:
                    cursor.execute(pred_sql)
                    if isinstance(gold_set, dict) and gold_set.get("kind") == "spider2_lite_csv_compare_v2_single_variant_fix":
                        from spider2_sqlite_data import compare_reference
                        pred_res = cursor.fetchmany(MAX_ROW_LIMIT + 1)
                        if len(pred_res) > MAX_ROW_LIMIT:
                            is_correct = 0
                        else:
                            is_correct = compare_reference(pred_res, [desc[0] for desc in cursor.description], gold_set)
                    else:
                        pred_res = cursor.fetchmany(MAX_ROW_LIMIT)
                        is_correct = 1 if set(pred_res) == gold_set else 0
                except Exception:
                    is_correct = 0
                finally:
                    timer.cancel()
                    
        res_list.append({
            "question_id": str(q_id),
            "generated_sql": pred_sql,
            "execution_result": is_correct
        })
        
    conn.close()
    return db_id, res_list

# ==========================================
# 5. Local model inference and evaluation
# ==========================================
def extract_sql_block(text: str) -> str:
    """Extract the last SELECT/WITH block and flatten SQL layout safely."""
    text = str(text).strip()
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1].strip()
    if "<think>" in text:
        return ""

    def clean_sql(sql):
        # Preserve quoted literals/identifiers. Remove comments before flattening,
        # so '-- comment\nFROM ...' cannot swallow the rest of the query.
        tokens = re.findall(
            r"'(?:''|[^'])*'|\"(?:\"\"|[^\"])*\"|`(?:``|[^`])*`|\[[^\]]*\]"
            r"|--[^\r\n]*|/\*.*?\*/|\s+|.", sql, flags=re.DOTALL)
        result = []
        for token in tokens:
            if token.isspace() or token.startswith(("--", "/*")):
                result.append(" ")
            else:
                result.append(token)
        return "".join(result).strip().rstrip(";").rstrip()

    blocks = re.findall(r"```(.*?)```", text, flags=re.DOTALL)
    for block in reversed(blocks):
        # sqlite must precede sql; use a boundary to avoid treating 'sqlite' as
        # a sql tag followed by the spurious SQL prefix 'ite'.
        candidate = re.sub(r"\A\s*(?:sqlite|sql)\b[ \t]*(?:\r?\n)?", "", block,
                           count=1, flags=re.IGNORECASE)
        sql = clean_sql(candidate)
        if re.match(r"(?i)(SELECT|WITH)\b", sql):
            return sql
    if blocks or "```" in text:
        return ""  # Never return an unfinished fence or surrounding explanation.
    sql = clean_sql(text)
    return sql if re.match(r"(?i)(SELECT|WITH)\b", sql) else ""

def run_model_inference_and_eval(model_name, model_path, bird_data, gold_cache):
    schema_fingerprint = prepare_schema_inputs(bird_data)
    csv_output_path = os.path.join(OUTPUT_DIR, f"{model_name}_results.csv")
    if result_csv_has_token_metrics(csv_output_path, schema_fingerprint,
                                    [question_key(row, i) for i, row in enumerate(bird_data)]):
        print(f"Existing execution records found: {csv_output_path}; skipping this model.")
        return csv_output_path
    
    print(f"\n>>> Loading model: {model_name} (path: {model_path})...")
    
    if LLM is None:
        raise RuntimeError("Local inference requires vllm; install it or use --results-only")
    model_kwargs = {"tensor_parallel_size": GPU_NUMS, "trust_remote_code": True,
                    "gpu_memory_utilization": 0.6}
    model_kwargs.update(MODELS_TO_RUN.get(model_name, {}).get("llm_kwargs", {}))
    llm = LLM(model=model_path, **model_kwargs)
    
    prompts =[]
    print("Building inference prompts...")
    for item in bird_data:
        db_id = item["db_id"]
        question = item["question"]
        evidence = item.get("evidence") or ""
        
        db_schema, _ = get_db_schema_and_path(db_id)
        schema_text = db_schema
        
        prompt = (
            f"You are a professional SQLite specialist. Given the database schema and a question, "
            f"generate the correct SQLite query.\n\n"
            f"Database Schema:\n{schema_text}\n\n"
            f"User Question: {question}\n"
            f"Hint/Evidence: {evidence}\n\n"
            f"Generate ONLY the executable SQL query inside ```sql... ``` block.\n"
            f"SQLite Query:"
        )
        prompts.append(prompt)

    sampling_params = SamplingParams(temperature=0.0, max_tokens=256)
    
    print(f"Starting vLLM batch inference for {len(prompts)} items...")
    model_cfg = MODELS_TO_RUN.get(model_name, {})
    if "enable_thinking" in model_cfg:
        # The switch is consumed by the chat template, not LLM/SamplingParams.
        outputs = llm.chat(
            [[{"role": "user", "content": prompt}] for prompt in prompts],
            sampling_params=sampling_params,
            chat_template_kwargs={"enable_thinking": model_cfg["enable_thinking"]},
        )
    else:
        outputs = llm.generate(prompts, sampling_params)
    
    generated_sqls =[]
    token_usages = []
    for output in outputs:
        candidate = output.outputs[0]
        raw_text = candidate.text
        if isinstance(raw_text, list):
            raw_text = raw_text[0] if len(raw_text) > 0 else ""
        text = str(raw_text).strip()
        generated_sqls.append(extract_sql_block(text))
        prompt_tokens = len(getattr(output, "prompt_token_ids", None) or [])
        completion_tokens = len(getattr(candidate, "token_ids", None) or [])
        token_usages.append({
            "prompt_tokens": int(prompt_tokens),
            "completion_tokens": int(completion_tokens),
            "total_tokens": int(prompt_tokens + completion_tokens),
            "token_source": "vllm_token_ids"
        })
        
    import gc
    import torch
    del llm
    gc.collect()
    torch.cuda.empty_cache()
    return save_eval_csv(
        model_name, bird_data, generated_sqls, gold_cache, token_usages=token_usages
    )

def save_eval_csv(model_name, bird_data, generated_sqls, gold_cache, token_usages=None):
    print("Comparing execution results...")
    db_group = defaultdict(list)
    for idx, item in enumerate(bird_data):
        db_group[item["db_id"]].append((idx, question_key(item, idx), generated_sqls[idx]))

    proc_num = min(cpu_count(), 8)
    pool = Pool(processes=proc_num)
    
    task_args = []
    for dbid, items in db_group.items():
        chunk_cache = {}
        for _, q_id, _ in items:
            blob = gold_cache.get(q_id)
            if blob is None and q_id.isdigit():
                blob = gold_cache.get(int(q_id))
            if blob is not None:
                chunk_cache[q_id] = blob
        task_args.append((dbid, items, DB_ROOT_DIR, chunk_cache))
        
    all_sub_results = []
    total_eval_dbs = len(task_args)
    completed_eval = 0
    
    for db_id, sublist in pool.imap_unordered(single_model_db_execute, task_args):
        all_sub_results.append(sublist)
        completed_eval += 1
        percent = (completed_eval / total_eval_dbs) * 100
        print(f"   Evaluation progress: [{completed_eval:3d}/{total_eval_dbs:3d}] ({percent:6.2f}%) | Database `{db_id}` complete", flush=True)
    pool.close()
    pool.join()

    records = [None] * len(bird_data)
    id_to_position = {question_key(item, i): i for i, item in enumerate(bird_data)}
    for sublist in all_sub_results:
        for row in sublist:
            pos = id_to_position[canonical_id(row["question_id"])]
            records[pos] = row

    if token_usages is None:
        token_usages = [None] * len(bird_data)
    if len(token_usages) != len(bird_data):
        raise ValueError("token_usages length does not match bird_data")
    for pos, row in enumerate(records):
        row["db_id"] = bird_data[pos]["db_id"]
        usage = token_usages[pos] or {}
        row["prompt_tokens"] = int(usage.get("prompt_tokens", 0))
        row["completion_tokens"] = int(usage.get("completion_tokens", 0))
        row["total_tokens"] = int(
            usage.get(
                "total_tokens",
                row["prompt_tokens"] + row["completion_tokens"]
            )
        )
        row["token_source"] = usage.get("token_source", "unavailable")

    csv_output_path = os.path.join(OUTPUT_DIR, f"{model_name}_results.csv")
    df_res = pd.DataFrame(records)
    # Remove the old marker first so an interrupted overwrite cannot reuse stale results.
    metadata_path = Path(str(csv_output_path) + ".schema.json")
    metadata_path.unlink(missing_ok=True)
    df_res.to_csv(csv_output_path, index=False)
    
    sql_json_dict = {str(r["question_id"]): r["generated_sql"] for r in records}
    with open(os.path.join(OUTPUT_DIR, f"{model_name}_sql_pred.json"), 'w', encoding='utf-8') as f_json:
        json.dump(sql_json_dict, f_json, indent=4, ensure_ascii=False)
    metadata_path.write_text(json.dumps({
        "schema_fingerprint": schema_input_fingerprint(bird_data),
        "schema_dir": str(Path(SCHEMA_DIR).expanduser().resolve()),
        "dataset_name": DATASET_NAME,
        "source_dataset_fingerprint": dataset_fingerprint(bird_data),
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    return csv_output_path

# ==========================================
# 6. API model inference and evaluation
# ==========================================
def run_api_inference_and_eval(model_name, api_cfg, bird_data, gold_cache):
    schema_fingerprint = prepare_schema_inputs(bird_data)
    csv_output_path = os.path.join(OUTPUT_DIR, f"{model_name}_results.csv")
    if result_csv_has_token_metrics(csv_output_path, schema_fingerprint,
                                    [question_key(row, i) for i, row in enumerate(bird_data)]):
        print(f"Complete valid CSV results already exist for API model {model_name}; skipping inference")
        return csv_output_path
    
    api_url = api_cfg["api_url"]
    api_key = os.environ.get(api_cfg.get("api_key_env", "")) or api_cfg.get("api_key")
    if not api_key:
        raise ValueError(f"{model_name}: configure api_key_env")
    api_model = api_cfg["model_name_api"]
    
    is_openai = api_cfg.get("api_style", "openai" if api_url.rstrip('/').endswith("chat/completions") else "anthropic") == "openai"
    if is_openai:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
    else:
        headers = {
            "x-api-key": api_key,
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01"
        }

    print(f"\n>>> Calling API model {model_name} | endpoint: {api_url}")
    full_prompts = []
    for item in bird_data:
        db_id = item["db_id"]
        question = item["question"]
        evidence = item.get("evidence") or ""
        db_schema, _ = get_db_schema_and_path(db_id)
        schema_text = db_schema
        prompt = (
            f"You are a professional SQLite specialist. Given the database schema and a question, "
            f"generate the correct SQLite query.\n\n"
            f"Database Schema:\n{schema_text}\n\n"
            f"User Question: {question}\n"
            f"Hint/Evidence: {evidence}\n\n"
            f"Generate ONLY the executable SQL query inside ```sql... ``` block.\n"
            f"SQLite Query:"
        )
        full_prompts.append(prompt)
    
    generated_sqls = []
    token_usages = []
    total = len(full_prompts)
    
    for start in range(0, total, API_BATCH_SIZE):
        batch_prompts = full_prompts[start:start+API_BATCH_SIZE]
        print(f"API inference progress: {start}/{total}")
        for p in batch_prompts:
            if is_openai:
                payload = {
                    "model": api_model,
                    "messages": [{"role": "user", "content": p}],
                    "temperature": 0.0,
                    "max_tokens": 256
                }
            else:
                payload = {
                    "model": api_model,
                    "max_tokens": 256,
                    "temperature": 0.0,
                    "messages": [{"role": "user", "content": p}]
                }
            try:
                resp = requests.post(api_url, headers=headers, json=payload, timeout=API_TIMEOUT)
                resp.raise_for_status()
                response_data = resp.json()
                if is_openai:
                    content = response_data["choices"][0]["message"]["content"]
                    usage = response_data.get("usage", {})
                    prompt_tokens = int(usage.get("prompt_tokens", 0) or 0)
                    completion_tokens = int(usage.get("completion_tokens", 0) or 0)
                    total_tokens = int(
                        usage.get("total_tokens", prompt_tokens + completion_tokens) or 0
                    )
                    token_source = "api_usage"
                else:
                    content = response_data["content"][0]["text"]
                    usage = response_data.get("usage", {})
                    prompt_tokens = int(usage.get("input_tokens", 0) or 0)
                    completion_tokens = int(usage.get("output_tokens", 0) or 0)
                    total_tokens = prompt_tokens + completion_tokens
                    token_source = "anthropic_usage"
                sql_out = extract_sql_block(content)
                generated_sqls.append(sql_out)
                token_usages.append({
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": total_tokens,
                    "token_source": token_source
                })
            except Exception as e:
                print(f"API request failed; recording empty SQL: {str(e)}")
                generated_sqls.append("")
                token_usages.append({
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                    "token_source": "request_failed"
                })
    
    return save_eval_csv(
        model_name, bird_data, generated_sqls, gold_cache, token_usages=token_usages
    )

# ==========================================
# 7. Aligned response matrix construction and 2PL-IRT calibration
# ==========================================
def build_aligned_response_matrix(bird_data, model_csv_paths):
    """
    Build a solver-by-item response matrix with shape M x N.
    M is the number of solvers and N is the number of items.
    """
    num_items = len(bird_data)
    model_names = list(model_csv_paths.keys())
    num_models = len(model_names)
    
    # Initialize item-by-model records, then transpose.
    response_matrix = np.zeros((num_items, num_models), dtype=np.int32)
    idx_to_row = {question_key(item, idx): idx for idx, item in enumerate(bird_data)}
    expected_ids = set(idx_to_row)
    
    for model_idx, model_name in enumerate(model_names):
        path = model_csv_paths[model_name]
        df = pd.read_csv(path, usecols=["question_id", "execution_result"], dtype={"question_id": str})
        df["question_id"] = df["question_id"].map(canonical_id)
        if df["question_id"].duplicated().any():
            duplicated = df.loc[df["question_id"].duplicated(), "question_id"].tolist()
            raise ValueError(
                f"Results for model {model_name} contain duplicate question_id values: {duplicated[:10]}"
            )
        supplied_ids = set(df["question_id"])
        missing_ids = expected_ids - supplied_ids
        extra_ids = supplied_ids - expected_ids
        if missing_ids or extra_ids:
            raise ValueError(
                f"Model {model_name} does not have a complete response vector: "
                f"{len(missing_ids)} missing items, {len(extra_ids)} extra items"
            )
        numeric_scores = pd.to_numeric(df["execution_result"], errors="raise")
        if not numeric_scores.isin([0, 1]).all():
            raise ValueError(f"execution_result for model {model_name} must contain only 0 or 1")
        row_ids = df["question_id"].map(idx_to_row).values
        scores = numeric_scores.astype(np.int32).values
        response_matrix[row_ids, model_idx] = scores
    # Transpose to solver-by-item layout for IRT.
    return response_matrix.T, model_names

# --------------------------
# Response augmentation 1: empirical sampling
# --------------------------
def sampling_augment(response_matrix, n_virtual):
    """
    Generate synthetic respondents from empirical item correctness rates.
    Implements the sampling augmentation method.
    """
    n_students, n_items = response_matrix.shape
    item_correct_rates = response_matrix.mean(axis=0)  # Mean correctness per item.
    
    virtual_students = np.zeros((n_virtual, n_items), dtype=np.int32)
    for i in range(n_virtual):
        for j in range(n_items):
            virtual_students[i, j] = 1 if np.random.random() < item_correct_rates[j] else 0
    return virtual_students

# --------------------------
# Response augmentation 2: variational autoencoder (VAE)
# --------------------------
class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim=32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        self.mu_layer = nn.Linear(64, latent_dim)
        self.logvar_layer = nn.Linear(64, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim),
            nn.Sigmoid()
        )
    
    def encode(self, x):
        h = self.encoder(x)
        return self.mu_layer(h), self.logvar_layer(h)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

def vae_augment(response_matrix, n_virtual, latent_dim=32, epochs=50, lr=1e-3):
    """
    Learn response distributions with a VAE and generate synthetic respondents.
    Uses a latent dimension of 32 by default.
    """
    n_students, n_items = response_matrix.shape
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    vae = VAE(n_items, latent_dim).to(device)
    optimizer = optim.Adam(vae.parameters(), lr=lr)
    data_tensor = torch.FloatTensor(response_matrix).to(device)
    dataset = torch.utils.data.TensorDataset(data_tensor)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    # Train the VAE.
    vae.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
            x = batch[0]
            recon, mu, logvar = vae(x)
            # Reconstruction loss plus Kullback-Leibler divergence.
            recon_loss = nn.BCELoss(reduction='sum')(recon, x)
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            loss = recon_loss + 0.5 * kl_loss  # beta-VAE，beta=0.5
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
    
    # Generate synthetic respondents.
    vae.eval()
    with torch.no_grad():
        z = torch.randn(n_virtual, latent_dim).to(device)
        generated_probs = vae.decode(z).cpu().numpy()
    # Sample binary responses from the predicted probabilities.
    virtual_students = (np.random.random(generated_probs.shape) < generated_probs).astype(np.int32)
    return virtual_students

# --------------------------
# Main 2PL-IRT calibration routine
# --------------------------
def iterative_filter_response_matrix(response_matrix):
    """Remove all-correct/all-incorrect rows and columns until both index sets stabilize."""
    matrix = np.asarray(response_matrix)
    if matrix.ndim != 2:
        raise ValueError("response_matrix must be two-dimensional")
    if matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError("response_matrix must not be empty")
    if not set(np.unique(matrix)).issubset({0, 1}):
        raise ValueError("response_matrix must contain only 0 and 1")

    student_indices = np.arange(matrix.shape[0], dtype=np.int64)
    item_indices = np.arange(matrix.shape[1], dtype=np.int64)
    iteration = 0

    while True:
        iteration += 1
        previous_student_count = len(student_indices)
        previous_item_count = len(item_indices)

        current = matrix[np.ix_(student_indices, item_indices)]
        informative_students = np.ptp(current, axis=1) > 0
        student_indices = student_indices[informative_students]
        if len(student_indices) == 0:
            raise RuntimeError("No valid respondents remain for IRT calibration after iterative filtering")

        current = matrix[np.ix_(student_indices, item_indices)]
        informative_items = np.ptp(current, axis=0) > 0
        item_indices = item_indices[informative_items]
        if len(item_indices) == 0:
            raise RuntimeError("No valid items remain for IRT calibration after iterative filtering")

        print(
            f"   Filtering iteration {iteration}: "
            f"{len(student_indices)} valid respondents, {len(item_indices)} valid items"
        )
        if (
            len(student_indices) == previous_student_count
            and len(item_indices) == previous_item_count
        ):
            break

    filtered = matrix[np.ix_(student_indices, item_indices)].astype(np.int32, copy=False)
    return filtered, student_indices, item_indices

def standardize_abilities_by_real_reference(
    theta_raw, n_real_students, epsilon=1e-6
):
    """
    Use retained real respondents to fix the ability axis location and scale,
    and apply the same affine transformation to real and synthetic respondents.

    Return standardized abilities, the real-respondent raw mean, and the stabilized scale.
    """
    if theta_raw.ndim != 1:
        raise ValueError("theta_raw must be a one-dimensional tensor")
    n_real_students = int(n_real_students)
    if n_real_students < 2 or n_real_students > theta_raw.numel():
        raise ValueError("Ability standardization requires at least two real respondents, not exceeding the total count")
    epsilon = float(epsilon)
    if not np.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("Ability standardization epsilon must be finite and positive")

    real_theta = theta_raw[:n_real_students]
    real_location = real_theta.mean()
    centered = theta_raw - real_location
    real_variance = centered[:n_real_students].pow(2).mean()
    real_scale = torch.sqrt(real_variance + epsilon)
    standardized = centered / real_scale
    return standardized, real_location, real_scale

def build_group_normalized_response_weights(
    n_real_students, n_synthetic_students, eta
):
    """Construct the group-normalized weights specified in paper section 3.1.2."""
    n_real_students = int(n_real_students)
    n_synthetic_students = int(n_synthetic_students)
    eta = float(eta)
    if n_real_students <= 0 or n_synthetic_students <= 0:
        raise ValueError("Both real and synthetic respondent counts must be positive")
    if not 0.5 < eta < 1.0:
        raise ValueError("The real-group weight eta must lie strictly within (0.5, 1)")

    total_students = n_real_students + n_synthetic_students
    real_weight = eta * total_students / n_real_students
    synthetic_weight = (1.0 - eta) * total_students / n_synthetic_students
    weights = np.concatenate([
        np.full(n_real_students, real_weight, dtype=np.float64),
        np.full(n_synthetic_students, synthetic_weight, dtype=np.float64),
    ])
    if not np.isclose(weights.sum(), total_students):
        raise RuntimeError("Group-normalized response weights do not sum to one")
    if not weights[:n_real_students].sum() > weights[n_real_students:].sum():
        raise RuntimeError("The real-response group does not receive the majority of the total weight")
    return weights

def run_irt_calibration_2pl(response_matrix, return_valid_student_indices=False):
    """
    Calibrate 2PL parameters using regularized maximum likelihood in PyTorch.
    - P(X_ij=1|theta_j)=1/(1+exp(-a_i*(theta_j-b_i))).
    - a_i is item discrimination.
    - b_i is item difficulty.
    - theta_j is respondent ability.
    - Return abilities only for retained real respondents, excluding synthetic respondents.
    """
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    warnings.filterwarnings("ignore", category=OptimizeWarning)

    print("\n" + "="*60)
    print(">>> Starting 2PL-IRT parameter calibration")
    print("="*60)
    
    n_students_raw, n_items_raw = response_matrix.shape
    print(f"Original real respondents: {n_students_raw}; original items: {n_items_raw}")
    
    # 1. Alternate filtering of degenerate respondents and items until stable.
    print("\n[Step 1] Iteratively filtering degenerate responses...")
    response_matrix, valid_student_indices, valid_item_indices = (
        iterative_filter_response_matrix(response_matrix)
    )
    n_real_students = response_matrix.shape[0]
    print(f"Filtering complete; retained real respondents: {n_real_students}")
    print(f"Filtering complete; retained items: {response_matrix.shape[1]}")
    
    if response_matrix.shape[0] < 2 or response_matrix.shape[1] < 10:
        raise RuntimeError("Insufficient valid responses for IRT calibration; add more models")
    
    # 2. Response augmentation with VAE and empirical sampling.
    print("\n[Step 2] Generating synthetic respondents for response augmentation...")
    n_vae = IRT_CONFIG["vae_virtual_students"]
    n_sample = IRT_CONFIG["sample_virtual_students"]
    
    vae_virtual = vae_augment(response_matrix, n_vae)
    sample_virtual = sampling_augment(response_matrix, n_sample)
    
    # Concatenate real and synthetic responses as an integer respondent-by-item matrix.
    augmented_matrix = np.vstack([response_matrix, vae_virtual, sample_virtual]).astype(np.int32)
    print(f"Augmented respondents: {augmented_matrix.shape[0]} (real {response_matrix.shape[0]} + VAE {n_vae} + sampled {n_sample})")
    
    # Validate response values.
    unique_vals = np.unique(augmented_matrix)
    if not set(unique_vals).issubset({0, 1}):
        raise ValueError(f"Invalid response values: {unique_vals}; only 0 and 1 are allowed")
    
    # 3. Estimate 2PL parameters.
    print("\n[Step 3] Estimating 2PL parameters with regularized MLE...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    y = torch.tensor(augmented_matrix, dtype=torch.float32, device=device)
    n_students, n_items = y.shape

    n_synthetic_students = n_students - n_real_students
    response_weights_np = build_group_normalized_response_weights(
        n_real_students,
        n_synthetic_students,
        IRT_CONFIG["2pl_real_group_weight_eta"],
    )

    row_mean = np.clip(augmented_matrix.mean(axis=1), 1e-4, 1 - 1e-4)
    # Compute column accuracy using respondent-group weights.
    col_mean = np.clip(
        np.average(augmented_matrix, axis=0, weights=response_weights_np),
        1e-4,
        1 - 1e-4,
    )
    theta_logit_init = np.log(row_mean / (1 - row_mean))
    ability_epsilon = float(IRT_CONFIG["2pl_ability_standardization_epsilon"])
    real_location_init = float(theta_logit_init[:n_real_students].mean())
    real_variance_init = float(np.mean(
        (theta_logit_init[:n_real_students] - real_location_init) ** 2
    ))
    real_scale_init = float(np.sqrt(real_variance_init + ability_epsilon))
    theta_init = (theta_logit_init - real_location_init) / real_scale_init
    # Transform empirical difficulty initializations to the real-respondent standardized scale.
    difficulty_init = (
        -np.log(col_mean / (1 - col_mean)) - real_location_init
    ) / real_scale_init

    theta_raw = nn.Parameter(torch.tensor(theta_init, dtype=torch.float32, device=device))
    difficulty_raw = nn.Parameter(torch.tensor(difficulty_init, dtype=torch.float32, device=device))
    discrimination_raw = nn.Parameter(torch.zeros(n_items, dtype=torch.float32, device=device))

    row_weight = torch.tensor(
        response_weights_np[:, None], dtype=torch.float32, device=device
    )

    min_a = float(IRT_CONFIG["2pl_min_discrimination"])
    max_a = float(IRT_CONFIG["2pl_max_discrimination"])
    optimizer = optim.Adam(
        [theta_raw, difficulty_raw, discrimination_raw],
        lr=float(IRT_CONFIG["2pl_learning_rate"])
    )

    for epoch in range(int(IRT_CONFIG["2pl_epochs"])):
        theta, _, _ = standardize_abilities_by_real_reference(
            theta_raw,
            n_real_students,
            epsilon=ability_epsilon,
        )
        difficulty_t = difficulty_raw
        discrimination_t = min_a + (max_a - min_a) * torch.sigmoid(discrimination_raw)
        logits = discrimination_t[None, :] * (theta[:, None] - difficulty_t[None, :])
        bce = nn.functional.binary_cross_entropy_with_logits(logits, y, reduction="none")
        # L_resp = sum_j omega_j sum_i BCE_ji / (|I| sum_j omega_j)
        likelihood_loss = (bce * row_weight).sum() / (
            n_items * row_weight.sum()
        )

        # Fix location and scale with real respondents; regularization limits augmented ability expansion.
        theta_prior = float(IRT_CONFIG["2pl_theta_regularization"]) * theta.pow(2).mean()
        difficulty_prior = float(
            IRT_CONFIG["2pl_difficulty_regularization"]
        ) * difficulty_t.pow(2).mean()
        discrimination_prior = float(
            IRT_CONFIG["2pl_log_discrimination_regularization"]
        ) * torch.log(discrimination_t).pow(2).mean()
        loss = likelihood_loss + theta_prior + difficulty_prior + discrimination_prior

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 200 == 0 or epoch == 0:
            print(f"   2PL epoch {epoch + 1}/{IRT_CONFIG['2pl_epochs']} | loss: {loss.item():.4f}")

    with torch.no_grad():
        ability_t, ability_location_t, ability_scale_t = (
            standardize_abilities_by_real_reference(
                theta_raw,
                n_real_students,
                epsilon=ability_epsilon,
            )
        )
        ability = ability_t.detach().cpu().numpy()
        ability_location = float(ability_location_t.detach().cpu().item())
        ability_scale = float(ability_scale_t.detach().cpu().item())
        difficulty = difficulty_raw.detach().cpu().numpy()
        discrimination = (
            min_a + (max_a - min_a) * torch.sigmoid(discrimination_raw)
        ).detach().cpu().numpy()

    ability = np.nan_to_num(ability, nan=0.0, posinf=5.0, neginf=-5.0)
    difficulty = np.nan_to_num(difficulty, nan=0.0, posinf=5.0, neginf=-5.0)
    discrimination = np.nan_to_num(discrimination, nan=1.0, posinf=max_a, neginf=min_a)
    
    # Synthetic responses stabilize item estimates. Absolute difficulty bounds use only real abilities.
    real_ability = ability[:n_real_students]
    print(f"Item difficulty range: [{difficulty.min():.3f}, {difficulty.max():.3f}]")
    print(f"Item discrimination range: [{discrimination.min():.3f}, {discrimination.max():.3f}]")
    if real_ability.size:
        print(f"Real respondent ability range: [{real_ability.min():.3f}, {real_ability.max():.3f}]")
        print(
            "Real respondent ability standardization check: "
            f"mean={real_ability.mean():.6f}, variance={np.mean(real_ability ** 2):.6f}, "
            f"raw_location={ability_location:.6f}, raw_scale={ability_scale:.6f}"
        )

    result = (difficulty, discrimination, valid_item_indices, real_ability)
    if return_valid_student_indices:
        return result + (valid_student_indices,)
    return result

# ==========================================
# 8. Pairwise difficulty ranker
# ==========================================
# --------------------------
# 8.1 Cached OpenAI embeddings
# --------------------------
def build_embedding_text(item, return_metadata=False):
    """Build the Text-to-SQL embedding input from the question and schema."""
    db_schema, _ = get_db_schema_and_path(item["db_id"])
    schema_text = db_schema
    prepared = prepare_embedding_text(
        item["question"], item.get("evidence") or "", schema_text,
        model=RANKER_CONFIG["embedding_model"],
        token_budget=RANKER_CONFIG["embedding_token_budget"])
    return (prepared.text, prepared) if return_metadata else prepared.text
def _exception_status_code(error):
    """Extract HTTP status codes from SDK and proxy exceptions."""
    status_code = getattr(error, "status_code", None)
    if status_code is None:
        response = getattr(error, "response", None)
        status_code = getattr(response, "status_code", None)
    try:
        return int(status_code) if status_code is not None else None
    except (TypeError, ValueError):
        return None


def _retry_after_seconds(error):
    """Read a numeric Retry-After header, or return None if unavailable."""
    response = getattr(error, "response", None)
    headers = getattr(response, "headers", None)
    if not headers:
        return None
    raw_value = headers.get("retry-after") or headers.get("Retry-After")
    try:
        value = float(raw_value)
    except (TypeError, ValueError):
        return None
    return value if np.isfinite(value) and value >= 0 else None


def _is_retryable_embedding_error(error):
    """Retry only rate limits, connection failures, timeouts, and transient server errors."""
    status_code = _exception_status_code(error)
    if status_code in {408, 409, 429} or (
        status_code is not None and 500 <= status_code <= 599
    ):
        # Quota and billing errors require intervention rather than retries.
        message = str(error).lower()
        non_retryable_markers = (
            "insufficient_quota",
            "billing",
            "quota exceeded",
            "\u914d\u989d\u4e0d\u8db3",  # Provider message indicating insufficient quota.
            "\u4f59\u989d\u4e0d\u8db3",  # Provider message indicating insufficient balance.
        )
        return not any(marker in message for marker in non_retryable_markers)

    error_name = type(error).__name__
    return error_name in {
        "RateLimitError",
        "InternalServerError",
        "APIConnectionError",
        "APITimeoutError",
        "TimeoutError",
        "ConnectionError",
    }


def _request_embedding_batch_with_retry(
    batch,
    model,
    *,
    client,
    batch_number,
    max_attempts,
    initial_delay,
    max_delay,
    jitter,
    request_timeout,
    sleep_fn=time.sleep,
    uniform_fn=random.uniform,
):
    """Request one embedding batch with bounded retries; raise when attempts are exhausted."""
    max_attempts = int(max_attempts)
    initial_delay = float(initial_delay)
    max_delay = float(max_delay)
    jitter = float(jitter)
    request_timeout = float(request_timeout)
    if max_attempts < 1:
        raise ValueError("embedding_max_attempts must be at least 1")
    if initial_delay < 0 or max_delay < initial_delay or jitter < 0:
        raise ValueError("Embedding retry settings require 0 <= initial_delay <= max_delay and jitter >= 0")
    if request_timeout <= 0:
        raise ValueError("embedding_request_timeout must be positive")

    for attempt in range(1, max_attempts + 1):
        try:
            return client.embeddings.create(
                input=batch,
                model=model,
                timeout=request_timeout,
            )
        except Exception as error:
            retryable = _is_retryable_embedding_error(error)
            if not retryable or attempt >= max_attempts:
                reason = "non-retryable error" if not retryable else "maximum attempts reached"
                raise RuntimeError(
                    f"Embedding batch {batch_number} failed; stopping: {reason}; "
                    f"attempt={attempt}/{max_attempts}，"
                    f"status={_exception_status_code(error)}，"
                    f"error={type(error).__name__}: {error}"
                ) from error

            exponential_delay = min(
                max_delay,
                initial_delay * (2 ** (attempt - 1)),
            )
            retry_after = _retry_after_seconds(error)
            base_delay = max(exponential_delay, retry_after or 0.0)
            wait_seconds = base_delay + uniform_fn(0.0, jitter)
            print(
                f"[Embedding retry] Batch {batch_number}, attempt {attempt}/{max_attempts} failed: "
                f"status={_exception_status_code(error)}，{type(error).__name__}: {error}\n"
                f"   Waiting {wait_seconds:.1f} seconds before attempt {attempt + 1}...",
                flush=True,
            )
            sleep_fn(wait_seconds)

def get_batch_embeddings(texts, model="text-embedding-3-large"):
    """Request embedding batches with bounded retries for transient errors."""
    all_embeddings = []
    batch_size = int(RANKER_CONFIG["embedding_batch_size"])
    if batch_size < 1:
        raise ValueError("embedding_batch_size must be a positive integer")
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        batch = [t.replace("\n", " ") for t in batch]
        batch_number = i // batch_size + 1
        response = _request_embedding_batch_with_retry(
            batch,
            model,
            client=get_openai_client(),
            batch_number=batch_number,
            max_attempts=RANKER_CONFIG["embedding_max_attempts"],
            initial_delay=RANKER_CONFIG["embedding_retry_initial_delay"],
            max_delay=RANKER_CONFIG["embedding_retry_max_delay"],
            jitter=RANKER_CONFIG["embedding_retry_jitter"],
            request_timeout=RANKER_CONFIG["embedding_request_timeout"],
        )
        batch_embeds = [np.array(d.embedding) for d in response.data]
        if len(batch_embeds) != len(batch):
            raise RuntimeError(
                f"Unexpected embedding count in batch {batch_number}: "
                f"requested {len(batch)} items, received {len(batch_embeds)}"
            )
        all_embeddings.extend(batch_embeds)
        print(f"Embedding progress: {min(i+batch_size, len(texts))}/{len(texts)}")
        time.sleep(0.1)
    return np.array(all_embeddings)


def embedding_input_fingerprint(texts, model):
    """Fingerprint the exact observable inputs and encoder used by section 3.2."""
    digest = hashlib.sha256()
    digest.update(str(model).encode("utf-8"))
    for text in texts:
        normalized = str(text).replace("\n", " ")
        encoded = normalized.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()

# --------------------------
# 8.2 Pairwise ranking dataset
# --------------------------
class PairwiseDataset(Dataset):
    def __init__(self, embeddings, difficulties, tolerance=0.0):
        """
        embeddings: (N, dim) item embeddings.
        difficulties: (N,) IRT difficulties.
        tolerance: retain both pair orientations only when |b_i-b_j| > tolerance.
        """
        tolerance = float(tolerance)
        if not np.isfinite(tolerance) or tolerance < 0:
            raise ValueError("Pairwise difficulty tolerance delta must be finite and nonnegative")

        self.pairs = []
        self.labels = []
        self.sample_weights = []
        n = len(difficulties)
        
        # Include both pair orientations to remove positional bias.
        for i in range(n):
            for j in range(i+1, n):
                d_i, d_j = difficulties[i], difficulties[j]
                difficulty_gap = abs(d_i - d_j)
                if difficulty_gap <= tolerance:
                    continue
                # Forward orientation.
                self.pairs.append((i, j))
                label = 1.0 if d_i > d_j else 0.0
                self.labels.append(label)
                w = np.sqrt(difficulty_gap)
                self.sample_weights.append(w)
                # Reverse orientation.
                self.pairs.append((j, i))
                self.labels.append(1.0 - label)
                self.sample_weights.append(w)
        
        self.embeddings = embeddings
        if not self.pairs:
            raise ValueError(
                "No pairs remain after tolerance filtering; reduce pair_tolerance or add training items"
            )
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        i, j = self.pairs[idx]
        e_i = torch.FloatTensor(self.embeddings[i])
        e_j = torch.FloatTensor(self.embeddings[j])
        
        # Concatenate features as [e_i || e_j || e_i-e_j || |e_i-e_j|].
        diff = e_i - e_j
        x = torch.cat([e_i, e_j, diff, torch.abs(diff)])
        return x, torch.FloatTensor([self.labels[idx]]), torch.FloatTensor([self.sample_weights[idx]])

# --------------------------
# 8.3 MLP ranker architecture
# --------------------------
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

# --------------------------
# 8.4 Ranker training
# --------------------------
def set_ranker_random_seed(seed):
    seed = int(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def compute_positive_pairwise_gaps(difficulties):
    """Enumerate the exact multiset of nonzero unordered difficulty gaps."""
    values = np.asarray(difficulties, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size < 2:
        raise ValueError("At least two finite IRT difficulties are required to estimate pairwise gaps")

    total_pairs = values.size * (values.size - 1) // 2
    _, duplicate_counts = np.unique(values, return_counts=True)
    zero_gap_pairs = int(np.sum(duplicate_counts * (duplicate_counts - 1) // 2))
    positive_pair_count = int(total_pairs - zero_gap_pairs)
    if positive_pair_count == 0:
        raise ValueError("All training items have identical IRT difficulty; pairwise supervision is unavailable")

    # Preallocate the exact size to avoid doubling peak memory during concatenation.
    positive_gaps = np.empty(positive_pair_count, dtype=np.float64)
    write_offset = 0
    for i in range(values.size - 1):
        row_gaps = np.abs(values[i] - values[i + 1:])
        row_gaps = row_gaps[row_gaps > 0]
        next_offset = write_offset + row_gaps.size
        positive_gaps[write_offset:next_offset] = row_gaps
        write_offset = next_offset
    if write_offset != positive_pair_count:
        raise RuntimeError("Nonzero unordered difficulty-gap count verification failed")
    return positive_gaps

def build_pair_tolerance_candidates(train_difficulties, config):
    """Build positive gap-quantile thresholds that satisfy the pair-retention constraint."""
    quantiles = config.get("pair_tolerance_quantiles", [])
    if not quantiles:
        raise ValueError("pair_tolerance_quantiles must not be empty")
    quantiles = sorted({float(value) for value in quantiles})
    if any(not 0 < value < 1 for value in quantiles):
        raise ValueError("Candidate quantile levels must lie strictly within (0, 1)")

    gaps = compute_positive_pairwise_gaps(train_difficulties)
    min_retention = float(config.get("pair_tolerance_min_retention", 0.0))
    if not 0 < min_retention <= 1:
        raise ValueError("pair_tolerance_min_retention must lie within (0, 1]")

    candidates = []
    seen_deltas = []
    for quantile in quantiles:
        delta = float(np.quantile(gaps, quantile))
        if not np.isfinite(delta) or delta <= 0:
            continue
        if any(np.isclose(delta, previous, rtol=1e-12, atol=1e-12) for previous in seen_deltas):
            continue
        retention = float(np.mean(gaps > delta))
        if retention < min_retention:
            continue
        candidates.append({
            "quantile": quantile,
            "delta": delta,
            "pair_retention": retention,
        })
        seen_deltas.append(delta)

    if not candidates:
        raise ValueError(
            "No positive tolerance satisfies the retention constraint; reduce "
            "pair_tolerance_min_retention or adjust candidate quantile levels"
        )
    return candidates, gaps

def split_ranker_train_validation(indices, difficulties, validation_size, seed):
    """Create training and fixed validation splits with at least two distinct difficulties each."""
    for offset in range(100):
        split_seed = int(seed) + offset
        train_idx, val_idx = train_test_split(
            indices, test_size=validation_size, random_state=split_seed
        )
        if (
            np.unique(difficulties[train_idx]).size >= 2
            and np.unique(difficulties[val_idx]).size >= 2
        ):
            return train_idx, val_idx, split_seed
    raise ValueError(
        "Cannot create training and validation splits with two distinct difficulties each; "
        "add more items or check for degenerate IRT difficulties"
    )

def create_ranker_model(input_dim, device):
    return DifficultyRanker(
        input_dim=input_dim,
        hidden_dims=RANKER_CONFIG["hidden_dims"],
        dropout=RANKER_CONFIG["dropout"],
    ).to(device)

def evaluate_ranker_on_fixed_pairs(model, data_loader, device):
    """Compute weighted loss and AUC on fixed validation pairs shared across candidates."""
    model.eval()
    weighted_loss_sum = 0.0
    weight_sum = 0.0
    all_predictions = []
    all_labels = []
    with torch.no_grad():
        for features, labels, weights in data_loader:
            features = features.to(device)
            labels = labels.to(device)
            weights = weights.to(device)
            predictions = model(features)
            bce = -(
                labels * torch.log(predictions + 1e-8)
                + (1 - labels) * torch.log(1 - predictions + 1e-8)
            )
            weighted_loss_sum += float((bce * weights).sum().item())
            weight_sum += float(weights.sum().item())
            all_predictions.extend(predictions.detach().cpu().numpy().flatten())
            all_labels.extend(labels.detach().cpu().numpy().flatten())

    if weight_sum <= 0:
        raise ValueError("The fixed validation pair set is empty")
    # Normalize by the sum of weights.
    validation_loss = weighted_loss_sum / weight_sum
    validation_auc = roc_auc_score(all_labels, all_predictions)
    return float(validation_loss), float(validation_auc)


def select_unique_difficulty_anchors(difficulties, anchor_count):
    values = np.asarray(difficulties, dtype=np.float64)
    count = min(int(anchor_count), len(values))
    if count < 2:
        raise ValueError("Monotonicity validation requires at least two anchors")
    targets = np.linspace(values.min(), values.max(), count)
    available = np.ones(len(values), dtype=bool)
    selected = []
    for target in targets:
        distances = np.abs(values - target)
        distances[~available] = np.inf
        index = int(np.argmin(distances))
        selected.append(index)
        available[index] = False
    return np.asarray(selected, dtype=np.int64)


def compute_validation_monotonic_violation(
    model, validation_embeddings, validation_difficulties, device, anchor_count
):
    """Measure average decreases in fixed-anchor scores as validation IRT difficulty increases."""
    anchor_indices = select_unique_difficulty_anchors(
        validation_difficulties, anchor_count
    )
    anchor_embeddings = np.asarray(validation_embeddings)[anchor_indices]
    anchor_difficulties = np.asarray(validation_difficulties)[anchor_indices]
    anchor_tensor = torch.as_tensor(anchor_embeddings, dtype=torch.float32, device=device)

    model.eval()
    scores = []
    with torch.no_grad():
        for index in range(len(anchor_tensor)):
            first = anchor_tensor[index].unsqueeze(0).expand(len(anchor_tensor), -1)
            difference = first - anchor_tensor
            features = torch.cat(
                [first, anchor_tensor, difference, torch.abs(difference)], dim=1
            )
            forward_probabilities = model(features).flatten()
            reverse_difference = anchor_tensor - first
            reverse_features = torch.cat(
                [anchor_tensor, first, reverse_difference, torch.abs(reverse_difference)],
                dim=1,
            )
            reverse_probabilities = model(reverse_features).flatten()
            probabilities = 0.5 * (
                forward_probabilities + 1.0 - reverse_probabilities
            )
            scores.append(float(probabilities.mean().item()))

    order = np.argsort(anchor_difficulties, kind="stable")
    sorted_difficulties = anchor_difficulties[order]
    sorted_scores = np.asarray(scores)[order]
    unique_difficulties, inverse = np.unique(sorted_difficulties, return_inverse=True)
    mean_scores = (
        np.bincount(inverse, weights=sorted_scores) / np.bincount(inverse)
    )
    if unique_difficulties.size < 2:
        return 0.0
    violations = np.maximum(0.0, mean_scores[:-1] - mean_scores[1:])
    return float(np.mean(violations))


def train_ranker_candidate(
    train_embeddings,
    train_difficulties,
    fixed_validation_loader,
    input_dim,
    device,
    tolerance,
    seed,
):
    set_ranker_random_seed(seed)
    train_dataset = PairwiseDataset(
        train_embeddings, train_difficulties, tolerance=tolerance
    )
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    train_loader = DataLoader(
        train_dataset,
        batch_size=RANKER_CONFIG["batch_size"],
        shuffle=True,
        generator=generator,
    )
    model = create_ranker_model(input_dim, device)
    optimizer = optim.Adam(model.parameters(), lr=RANKER_CONFIG["learning_rate"])

    best_state = None
    best_validation_loss = float("inf")
    best_validation_auc = float("nan")
    best_epoch = 0
    for epoch in range(int(RANKER_CONFIG["epochs"])):
        model.train()
        training_loss_sum = 0.0
        training_weight_sum = 0.0
        for features, labels, weights in train_loader:
            features = features.to(device)
            labels = labels.to(device)
            weights = weights.to(device)
            predictions = model(features)
            bce = -(
                labels * torch.log(predictions + 1e-8)
                + (1 - labels) * torch.log(1 - predictions + 1e-8)
            )
            batch_weight_sum = weights.sum()
            if batch_weight_sum.item() <= 0:
                raise RuntimeError("Training-batch pair weights must have a positive sum")
            weighted_bce_sum = (bce * weights).sum()
            loss = weighted_bce_sum / batch_weight_sum
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            training_loss_sum += float(weighted_bce_sum.detach().item())
            training_weight_sum += float(batch_weight_sum.detach().item())

        validation_loss, validation_auc = evaluate_ranker_on_fixed_pairs(
            model, fixed_validation_loader, device
        )
        average_training_loss = training_loss_sum / max(training_weight_sum, 1e-12)
        print(
            f"      epoch {epoch + 1}/{RANKER_CONFIG['epochs']} | "
            f"train={average_training_loss:.4f} | "
            f"fixed-val={validation_loss:.4f} | AUC={validation_auc:.4f}"
        )
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_validation_auc = validation_auc
            best_epoch = epoch + 1
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

    if best_state is None:
        raise RuntimeError("The candidate ranker did not produce a valid model state")
    model.load_state_dict(best_state)
    return model, {
        "validation_loss": float(best_validation_loss),
        "validation_auc": float(best_validation_auc),
        "best_epoch": int(best_epoch),
        "retained_ordered_pair_count": int(len(train_dataset)),
    }


def train_final_ranker(
    embeddings, difficulties, input_dim, device, tolerance, epochs, seed
):
    set_ranker_random_seed(seed)
    dataset = PairwiseDataset(embeddings, difficulties, tolerance=tolerance)
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    loader = DataLoader(
        dataset,
        batch_size=RANKER_CONFIG["batch_size"],
        shuffle=True,
        generator=generator,
    )
    model = create_ranker_model(input_dim, device)
    optimizer = optim.Adam(model.parameters(), lr=RANKER_CONFIG["learning_rate"])
    for epoch in range(int(epochs)):
        model.train()
        total_loss = 0.0
        total_weight = 0.0
        for features, labels, weights in loader:
            features = features.to(device)
            labels = labels.to(device)
            weights = weights.to(device)
            predictions = model(features)
            bce = -(
                labels * torch.log(predictions + 1e-8)
                + (1 - labels) * torch.log(1 - predictions + 1e-8)
            )
            batch_weight_sum = weights.sum()
            if batch_weight_sum.item() <= 0:
                raise RuntimeError("Final-training pair weights must have a positive sum")
            weighted_bce_sum = (bce * weights).sum()
            loss = weighted_bce_sum / batch_weight_sum
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += float(weighted_bce_sum.detach().item())
            total_weight += float(batch_weight_sum.detach().item())
        print(
            f"   Final training epoch {epoch + 1}/{epochs} | "
            f"loss={total_loss / max(total_weight, 1e-12):.4f}"
        )
    return model, int(len(dataset))


RANKER_TRAINING_CONFIG_KEYS = (
    "embedding_model",
    "embedding_dim",
    "embedding_token_budget",
    "hidden_dims",
    "dropout",
    "batch_size",
    "epochs",
    "learning_rate",
    "pair_tolerance_mode",
    "pair_tolerance_quantiles",
    "pair_tolerance_min_retention",
    "tolerance_selection_seed",
    "tolerance_selection_repeats",
    "tolerance_monotonic_weight",
    "tolerance_stability_weight",
    "tolerance_anchor_count",
    "pair_tolerance",
    "activation",
)


def _array_sha256(array, dtype):
    normalized = np.ascontiguousarray(np.asarray(array, dtype=dtype))
    digest = hashlib.sha256()
    digest.update(str(normalized.dtype).encode("ascii"))
    digest.update(json.dumps(normalized.shape).encode("ascii"))
    digest.update(normalized.tobytes(order="C"))
    return digest.hexdigest(), normalized


def build_ranker_training_contract(
    embeddings,
    difficulties,
    valid_item_indices=None,
    config=None,
):
    """Build a persistent signature of all inputs that affect ranker training."""
    config = RANKER_CONFIG if config is None else config
    embedding_hash, normalized_embeddings = _array_sha256(embeddings, np.float32)
    difficulty_hash, normalized_difficulties = _array_sha256(difficulties, np.float64)
    if normalized_embeddings.ndim != 2 or normalized_difficulties.ndim != 1:
        raise ValueError("The ranker signature requires 2-D embeddings and 1-D difficulties")
    if len(normalized_embeddings) != len(normalized_difficulties):
        raise ValueError("Embedding and difficulty counts differ in the ranker signature")

    if valid_item_indices is None:
        valid_item_indices = np.arange(len(normalized_difficulties), dtype=np.int64)
    item_hash, normalized_item_indices = _array_sha256(valid_item_indices, np.int64)
    if normalized_item_indices.ndim != 1:
        raise ValueError("valid_item_indices must be one-dimensional")
    if len(normalized_item_indices) != len(normalized_difficulties):
        raise ValueError("The valid-item index count must match the ranker training item count")

    config_snapshot = {
        key: config.get(key)
        for key in RANKER_TRAINING_CONFIG_KEYS
    }
    config_json = json.dumps(
        config_snapshot,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    config_hash = hashlib.sha256(config_json.encode("utf-8")).hexdigest()
    combined = hashlib.sha256()
    for value in (
        RANKER_METHOD_VERSION,
        embedding_hash,
        difficulty_hash,
        item_hash,
        config_hash,
    ):
        combined.update(value.encode("ascii"))

    return {
        "method_version": RANKER_METHOD_VERSION,
        "training_fingerprint": combined.hexdigest(),
        "embedding_sha256": embedding_hash,
        "difficulty_sha256": difficulty_hash,
        "valid_item_indices_sha256": item_hash,
        "ranker_config_sha256": config_hash,
        "training_item_count": int(len(normalized_difficulties)),
        "embedding_shape": [int(value) for value in normalized_embeddings.shape],
        "ranker_config": config_snapshot,
    }


def _ranker_metadata_path(save_path):
    return os.path.splitext(save_path)[0] + "_metadata.json"


def _load_torch_checkpoint_cpu(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        # Support older PyTorch versions without weights_only.
        return torch.load(path, map_location="cpu")


def ranker_checkpoint_is_reusable(save_path, expected_contract):
    """Validate the checkpoint, metadata, and current training signature; return (reusable, reason)."""
    metadata_path = _ranker_metadata_path(save_path)
    if not os.path.exists(save_path):
        return False, "Ranker checkpoint does not exist"
    if not os.path.exists(metadata_path):
        return False, "Ranker metadata does not exist"

    try:
        with open(metadata_path, "r", encoding="utf-8") as handle:
            sidecar_metadata = json.load(handle)
    except Exception as error:
        return False, f"Cannot read ranker metadata: {type(error).__name__}: {error}"

    for key, expected_value in expected_contract.items():
        if sidecar_metadata.get(key) != expected_value:
            return False, f"Training signature mismatch: {key}"

    try:
        checkpoint = _load_torch_checkpoint_cpu(save_path)
    except Exception as error:
        return False, f"Cannot read ranker checkpoint: {type(error).__name__}: {error}"
    if not isinstance(checkpoint, dict):
        return False, "Ranker checkpoint is not a dictionary"
    checkpoint_metadata = checkpoint.get("metadata")
    state_dict = checkpoint.get("model_state_dict")
    if not isinstance(checkpoint_metadata, dict) or not isinstance(state_dict, dict):
        return False, "Ranker checkpoint lacks model state or training metadata"
    for key, expected_value in expected_contract.items():
        if checkpoint_metadata.get(key) != expected_value:
            return False, f"Checkpoint training signature mismatch: {key}"

    try:
        config = expected_contract["ranker_config"]
        input_dim = int(config["embedding_dim"]) * 4
        verification_model = DifficultyRanker(
            input_dim=input_dim,
            hidden_dims=config["hidden_dims"],
            dropout=float(config["dropout"]),
        )
        verification_model.load_state_dict(state_dict, strict=True)
        if any(
            not torch.isfinite(parameter).all().item()
            for parameter in verification_model.state_dict().values()
        ):
            return False, "Ranker parameters contain non-finite values"
    except Exception as error:
        return False, f"Ranker architecture or parameter validation failed: {type(error).__name__}: {error}"
    return True, "Training inputs, configuration, method version, and checkpoint match"


def train_ranker(
    embeddings,
    difficulties,
    save_path=RANKER_MODEL_PATH,
    training_contract=None,
):
    """
    Select a positive gap-quantile threshold on fixed validation pairs, then fit the final ranker.
    """
    print("\n" + "="*60)
    print(">>> Training the pairwise difficulty ranker")
    print("="*60)

    embeddings = np.asarray(embeddings, dtype=np.float32)
    difficulties = np.asarray(difficulties, dtype=np.float64)
    if embeddings.ndim != 2 or difficulties.ndim != 1:
        raise ValueError("embeddings must be two-dimensional and difficulties must be one-dimensional")
    if len(embeddings) != len(difficulties):
        raise ValueError("Embedding and IRT difficulty counts differ")
    if len(difficulties) < 4:
        raise ValueError("Tolerance selection requires at least four training items")
    if not np.all(np.isfinite(embeddings)) or not np.all(np.isfinite(difficulties)):
        raise ValueError("Embeddings or IRT difficulties contain non-finite values")
    expected_embedding_dim = int(RANKER_CONFIG["embedding_dim"])
    if embeddings.shape[1] != expected_embedding_dim:
        raise ValueError(
            f"Embedding dimension is {embeddings.shape[1]}, but the configuration specifies {expected_embedding_dim}"
        )

    input_dim = expected_embedding_dim * 4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    selection_seed = int(RANKER_CONFIG.get("tolerance_selection_seed", 42))
    indices = np.arange(len(difficulties))
    validation_size = max(2, int(round(0.2 * len(indices))))
    validation_size = min(validation_size, len(indices) - 2)
    train_idx, val_idx, split_seed = split_ranker_train_validation(
        indices, difficulties, validation_size, selection_seed
    )

    # All candidates share the same validation split and pairs, independent of delta.
    fixed_validation_dataset = PairwiseDataset(
        embeddings[val_idx], difficulties[val_idx], tolerance=0.0
    )
    fixed_validation_loader = DataLoader(
        fixed_validation_dataset,
        batch_size=RANKER_CONFIG["batch_size"],
        shuffle=False,
    )

    mode = str(RANKER_CONFIG.get("pair_tolerance_mode", "quantile_validation"))
    if mode == "quantile_validation":
        candidates, train_gaps = build_pair_tolerance_candidates(
            difficulties[train_idx], RANKER_CONFIG
        )
    elif mode == "fixed":
        fixed_delta = float(RANKER_CONFIG.get("pair_tolerance", 0.0))
        if not np.isfinite(fixed_delta) or fixed_delta < 0:
            raise ValueError("pair_tolerance must be finite and nonnegative in fixed mode")
        train_gaps = compute_positive_pairwise_gaps(difficulties[train_idx])
        candidates = [{
            "quantile": None,
            "delta": fixed_delta,
            "pair_retention": float(np.mean(train_gaps > fixed_delta)),
        }]
    else:
        raise ValueError("pair_tolerance_mode must be quantile_validation or fixed")

    print(
        f"Tolerance selection: training items={len(train_idx)}, validation items={len(val_idx)}, "
        f"fixed ordered validation pairs={len(fixed_validation_dataset)}, candidates={len(candidates)}"
    )

    repeats = int(RANKER_CONFIG.get("tolerance_selection_repeats", 1))
    if repeats < 1:
        raise ValueError(
            "The stability term requires at least one independent fit; "
        )
    monotonic_weight = float(RANKER_CONFIG.get("tolerance_monotonic_weight", 0.0))
    stability_weight = float(RANKER_CONFIG.get("tolerance_stability_weight", 0.0))
    anchor_count = int(RANKER_CONFIG.get("tolerance_anchor_count", 32))
    candidate_reports = []

    for candidate_index, candidate in enumerate(candidates):
        delta = float(candidate["delta"])
        print(
            f"\n   [Candidate {candidate_index + 1}/{len(candidates)}] "
            f"quantile={candidate['quantile']}，delta={delta:.6g}，"
            f"exact retention={candidate['pair_retention']:.4f}"
        )
        repeat_reports = []
        for repeat in range(repeats):
            repeat_seed = selection_seed + repeat
            candidate_model, metrics = train_ranker_candidate(
                embeddings[train_idx],
                difficulties[train_idx],
                fixed_validation_loader,
                input_dim,
                device,
                tolerance=delta,
                seed=repeat_seed,
            )
            metrics["monotonic_violation"] = compute_validation_monotonic_violation(
                candidate_model,
                embeddings[val_idx],
                difficulties[val_idx],
                device,
                anchor_count,
            )
            metrics["seed"] = int(repeat_seed)
            repeat_reports.append(metrics)
            del candidate_model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        validation_losses = np.asarray(
            [report["validation_loss"] for report in repeat_reports]
        )
        monotonic_violations = np.asarray(
            [report["monotonic_violation"] for report in repeat_reports]
        )
        objective = (
            float(validation_losses.mean())
            + monotonic_weight * float(monotonic_violations.mean())
            + stability_weight * float(validation_losses.std(ddof=0))
        )
        report = {
            **candidate,
            "mean_validation_loss": float(validation_losses.mean()),
            "mean_validation_auc": float(np.mean([
                item["validation_auc"] for item in repeat_reports
            ])),
            "mean_monotonic_violation": float(monotonic_violations.mean()),
            "validation_loss_std": float(validation_losses.std(ddof=0)),
            "mean_best_epoch": float(np.mean([
                item["best_epoch"] for item in repeat_reports
            ])),
            "selection_objective": float(objective),
            "repeats": repeat_reports,
        }
        candidate_reports.append(report)
        print(
            f"   Candidate summary: fixed-val={report['mean_validation_loss']:.4f}, "
            f"AUC={report['mean_validation_auc']:.4f}，"
            f"monotonicity violation={report['mean_monotonic_violation']:.6f}, "
            f"objective={objective:.4f}"
        )

    selected = min(candidate_reports, key=lambda item: item["selection_objective"])
    selected_quantile = selected["quantile"]
    selection_delta = float(selected["delta"])
    final_epochs = max(1, int(round(selected["mean_best_epoch"])))

    # Select the quantile on validation data, then recompute delta from all training gaps.
    if mode == "quantile_validation":
        full_gaps = compute_positive_pairwise_gaps(difficulties)
        final_delta = float(np.quantile(full_gaps, selected_quantile))
        final_retention = float(np.mean(full_gaps > final_delta))
    else:
        final_delta = selection_delta
        final_retention = float(np.mean(train_gaps > final_delta))
    min_retention = float(RANKER_CONFIG.get("pair_tolerance_min_retention", 0.0))
    if final_retention < min_retention:
        raise ValueError(
            "The recomputed tolerance on all training items violates the minimum pair-retention constraint; "
            "reduce the candidate quantile levels or pair_tolerance_min_retention"
        )

    print(
        f"\nSelected tolerance: quantile={selected_quantile}, "
        f"selection_delta={selection_delta:.6g}，final_delta={final_delta:.6g}，"
        f"full-training estimated retention={final_retention:.4f}"
    )
    final_model, final_pair_count = train_final_ranker(
        embeddings,
        difficulties,
        input_dim,
        device,
        tolerance=final_delta,
        epochs=final_epochs,
        seed=selection_seed,
    )
    if training_contract is None:
        training_contract = build_ranker_training_contract(
            embeddings,
            difficulties,
        )
    metadata = {
        **training_contract,
        "method_version": RANKER_METHOD_VERSION,
        "method": mode,
        "selected_quantile": selected_quantile,
        "selection_delta": selection_delta,
        "final_delta": final_delta,
        "final_pair_retention": final_retention,
        "final_ordered_pair_count": final_pair_count,
        "final_epochs": final_epochs,
        "selection_seed": selection_seed,
        "split_seed": split_seed,
        "fixed_validation_ordered_pair_count": int(len(fixed_validation_dataset)),
        "pair_gap_distribution": "exact_nonzero_unordered_multiset",
        "pair_loss_weight": "sqrt_absolute_irt_gap",
        "candidate_reports": candidate_reports,
        "selection_objective": {
            "validation_loss_weight": 1.0,
            "monotonic_violation_weight": monotonic_weight,
            "validation_loss_std_weight": stability_weight,
        },
    }
    checkpoint_temp_path = save_path + ".tmp"
    torch.save(
        {"model_state_dict": final_model.state_dict(), "metadata": metadata},
        checkpoint_temp_path,
    )
    os.replace(checkpoint_temp_path, save_path)
    metadata_path = _ranker_metadata_path(save_path)
    metadata_temp_path = metadata_path + ".tmp"
    with open(metadata_temp_path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False)
    os.replace(metadata_temp_path, metadata_path)

    print(f"\nRanker training complete; checkpoint saved to: {save_path}")
    print(f"Tolerance selection metadata saved to: {metadata_path}")
    return final_model

# --------------------------
# 8.5 Pairwise ranker inference
# --------------------------
def predict_difficulty_pair(model, embed_a, embed_b, device=None):
    """
    Estimate whether item a is more difficult than item b.
    Return the probability p(a > b).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    
    e_a = torch.FloatTensor(embed_a).to(device)
    e_b = torch.FloatTensor(embed_b).to(device)
    diff_ab = e_a - e_b
    diff_ba = e_b - e_a
    x_ab = torch.cat([e_a, e_b, diff_ab, torch.abs(diff_ab)]).unsqueeze(0)
    x_ba = torch.cat([e_b, e_a, diff_ba, torch.abs(diff_ba)]).unsqueeze(0)
    
    with torch.no_grad():
        p_ab = model(x_ab).item()
        p_ba = model(x_ba).item()
    # Symmetrize predictions so that p(a>b) + p(b>a) = 1.
    return 0.5 * (p_ab + 1.0 - p_ba)

# ==========================================
# 9. End-to-end entry point
# ==========================================
def load_inference_models(path):
    raw = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    entries, defaults = raw.get("models", raw), raw.get("defaults", {})
    if not isinstance(entries, dict) or not entries:
        raise ValueError("Model config needs a nonempty model mapping")
    models = {}
    for name, entry in entries.items():
        cfg = {**defaults, **entry}
        cfg["llm_kwargs"] = {**defaults.get("llm_kwargs", {}), **entry.get("llm_kwargs", {})}
        cfg.setdefault("type", "api" if cfg.get("backend") == "openai-compatible" else "local")
        if cfg["type"] == "api" and cfg.get("backend") == "openai-compatible":
            cfg.update(api_url=cfg["base_url"].rstrip('/') + '/chat/completions',
                       model_name_api=cfg["model"], api_style="openai")
        if cfg["type"] not in ("local", "api"):
            raise ValueError(f"{name}: unsupported type")
        if cfg["type"] == "local" and not cfg.get("model_path"):
            raise ValueError(f"{name}: missing model_path")
        models[name] = cfg
    return models


def validate_existing_results(path, expected_ids, dataset_name, source_fingerprint):
    # source_fingerprint is retained for compatibility, but is not compared for CSVs.
    frame = pd.read_csv(path, dtype={"question_id": str})
    required = {"question_id", "execution_result", *TOKEN_METRIC_COLUMNS}
    if not required.issubset(frame.columns):
        raise ValueError(f"{path}: missing columns {required - set(frame.columns)}")
    ids = frame["question_id"].map(canonical_id)
    if ids.duplicated().any() or set(ids) != set(expected_ids):
        raise ValueError(f"{path}: incomplete/duplicate training question IDs")
    if not frame["execution_result"].isin([0, 1]).all():
        raise ValueError(f"{path}: execution_result must be 0/1")
    usage = frame[sorted(TOKEN_METRIC_COLUMNS)]
    if (usage.isnull().any().any() or not (usage >= 0).all().all()
            or not (frame["total_tokens"] > 0).all()
            or not (frame["prompt_tokens"] + frame["completion_tokens"] == frame["total_tokens"]).all()):
        raise ValueError(f"{path}: incomplete measured token usage")
    metadata = Path(str(path) + ".schema.json")
    if metadata.is_file():
        meta = json.loads(metadata.read_text(encoding="utf-8-sig"))
        if meta.get("dataset_name", dataset_name) != dataset_name:
            raise ValueError(f"{path}: dataset mismatch")


def main(argv=None):
    global DATASET_NAME, BIRD_TRAIN_JSON, DB_ROOT_DIR, SCHEMA_DIR, TABLES_JSON
    global SCHEMA_FILE_PATTERN, OUTPUT_DIR, ARTIFACT_DIR, FINAL_IRT_JSON
    global GOLD_CACHE_FILE, EMBEDDING_CACHE_FILE, RANKER_MODEL_PATH, MODELS_TO_RUN, GPU_NUMS
    global MAX_ROW_LIMIT, SQL_TIMEOUT_SEC
    parser = argparse.ArgumentParser(description="BIRD/Spider 2PL training and pairwise ranker")
    parser.add_argument("--dataset-name", choices=("bird", "spider", "spider2_sqlite"), default="bird")
    parser.add_argument("--spider2-source", help="Native Spider2-Lite ZIP or extracted directory")
    parser.add_argument("--spider2-train-ids", help="Custom training IDs: local* IDs, JSON list or one per line")
    parser.add_argument("--spider2-use-all-as-train", action="store_true", help="Explicitly use all local* tasks as custom history; never evaluate on these same tasks")
    parser.add_argument("--spider2-prepare-only", action="store_true", help="Inspect/export normalized SQLite records and DDL without DBs, models, or APIs")
    parser.add_argument("--train-json", default=BIRD_TRAIN_JSON)
    parser.add_argument("--db-root", help="SQLite root for training execution labels; flat or db_id/db_id.sqlite")
    schemas = parser.add_mutually_exclusive_group()
    schemas.add_argument("--schema-dir", help="UTF-8 <db_id>.txt or <db_id>_schema.txt directory")
    schemas.add_argument("--tables-json", help="Official tables.json; use the same representation for positioning")
    parser.add_argument("--schema-file-pattern", choices=("auto", *SCHEMA_FILE_PATTERNS), default="auto")
    parser.add_argument("--output-dir", required=True, help="Directory for generated calibration artifacts")
    parser.add_argument("--model-results-dir", help="Override artifact_root/model_results")
    parser.add_argument("--models-config", help="JSON model mapping: existing local/api or direct-SQL models config")
    parser.add_argument("--training-config", help="Optional JSON with irt and ranker config overrides")
    parser.add_argument("--results-only", action="store_true", help="Use complete measured training CSVs without inference")
    parser.add_argument("--sql-timeout", type=float, help="Per-SQL seconds; Spider2 default 60, legacy default unchanged")
    parser.add_argument("--max-result-rows", type=int, help="Result row budget; Spider2 default 100000, legacy default unchanged")
    parser.add_argument("--gpu-ids", help="CUDA_VISIBLE_DEVICES for this run")
    parser.add_argument("--num-gpus", type=int, default=GPU_NUMS)
    parser.add_argument("--dry-run", action="store_true", help="Check dataset/schema/DBs without training or API calls")
    args = parser.parse_args(argv)
    if args.dataset_name != "spider2_sqlite" and any((args.spider2_source, args.spider2_train_ids, args.spider2_use_all_as_train, args.spider2_prepare_only)):
        parser.error("Spider2 arguments require --dataset-name spider2_sqlite")
    if args.dataset_name == "spider2_sqlite" and not args.spider2_prepare_only and not args.db_root:
        parser.error("Spider2 ZIP contains no SQLite files: supply downloaded local database directory via --db-root")
    if args.spider2_prepare_only and not args.spider2_source:
        parser.error("--spider2-prepare-only requires --spider2-source")
    DATASET_NAME = args.dataset_name
    SQL_TIMEOUT_SEC = args.sql_timeout if args.sql_timeout is not None else (60.0 if DATASET_NAME == "spider2_sqlite" else DEFAULT_SQL_TIMEOUT_SEC)
    MAX_ROW_LIMIT = args.max_result_rows if args.max_result_rows is not None else (100000 if DATASET_NAME == "spider2_sqlite" else DEFAULT_MAX_ROW_LIMIT)
    if not np.isfinite(SQL_TIMEOUT_SEC) or SQL_TIMEOUT_SEC <= 0 or MAX_ROW_LIMIT <= 0:
        parser.error("--sql-timeout and --max-result-rows must be finite positive values")
    BIRD_TRAIN_JSON, DB_ROOT_DIR = args.train_json, args.db_root or DB_ROOT_DIR
    SCHEMA_DIR, TABLES_JSON = args.schema_dir or SCHEMA_DIR, args.tables_json
    SCHEMA_FILE_PATTERN = args.schema_file_pattern
    ARTIFACT_DIR = str(Path(args.output_dir).resolve())
    OUTPUT_DIR = str(Path(args.model_results_dir or Path(ARTIFACT_DIR) / "model_results").resolve())
    FINAL_IRT_JSON = str(Path(ARTIFACT_DIR) / f"{DATASET_NAME}_train_with_irt_2pl_mle.json")
    GOLD_CACHE_FILE = str(Path(ARTIFACT_DIR) / "gold_results_cache.pkl")
    EMBEDDING_CACHE_FILE = str(Path(ARTIFACT_DIR) / "2PL_embedding_cache.pkl")
    RANKER_MODEL_PATH = str(Path(ARTIFACT_DIR) / "irt_2pl_difficulty_ranker.pth")
    if args.num_gpus < 1:
        parser.error("--num-gpus must be positive")
    GPU_NUMS = args.num_gpus
    if args.gpu_ids:
        if len(args.gpu_ids.split(',')) != GPU_NUMS:
            parser.error("--num-gpus must match --gpu-ids")
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
    if args.models_config:
        MODELS_TO_RUN = load_inference_models(args.models_config)
    elif not args.results_only and not args.dry_run and not args.spider2_prepare_only:
        parser.error("Supply --models-config or --results-only")
    if args.training_config:
        config = json.loads(Path(args.training_config).read_text(encoding="utf-8-sig"))
        IRT_CONFIG.update(config.get("irt", {}))
        RANKER_CONFIG.update(config.get("ranker", {}))
    print("="*70)
    print(f" {DATASET_NAME.upper()} difficulty-aware Text-to-SQL pipeline (2PL + ranker)")
    print("="*70)
    
    # 1. Load source data.
    if DATASET_NAME == "spider2_sqlite" and args.spider2_source:
        from spider2_sqlite_data import load_spider2_sqlite, read_train_ids
        from dataset_utils import normalize_dataset
        if args.schema_dir or args.tables_json:
            parser.error("Native Spider2 input uses its DDL.csv; omit --schema-dir/--tables-json")
        rows, schema_texts, source_manifest = load_spider2_sqlite(
            args.spider2_source, read_train_ids(args.spider2_train_ids),
            args.spider2_use_all_as_train, args.spider2_prepare_only)
        bird_train_data = normalize_dataset(rows, DATASET_NAME, require_gold=True,
                                            split=None if args.spider2_prepare_only else "train")
        SCHEMA_DIR, TABLES_JSON = str(Path(ARTIFACT_DIR) / "schemas"), None
        if not args.dry_run:
            Path(SCHEMA_DIR).mkdir(parents=True, exist_ok=True)
            for db_id, text in schema_texts.items():
                path = Path(SCHEMA_DIR) / f"{db_id}_schema.txt"
                if not path.exists() or path.read_text(encoding="utf-8") != text:
                    path.write_text(text, encoding="utf-8")
            write_json(Path(ARTIFACT_DIR) / "spider2_sqlite_source_manifest.json", source_manifest)
            write_json(Path(ARTIFACT_DIR) / "spider2_sqlite_normalized.json", bird_train_data)
        if args.spider2_prepare_only:
            print(json.dumps(source_manifest, ensure_ascii=False, indent=2))
            print("Input preparation complete; no models loaded, API calls made, or training started.")
            return 0
        if args.spider2_use_all_as_train:
            print("Warning: all SQLite items are used as a custom historical training pool and cannot also serve as an independent test set.")
        if args.dry_run:
            # Avoid materializing schemas on a read-only preflight.
            _SCHEMA_TEXTS.clear()
            _SCHEMA_TEXTS.update(schema_texts)
    else:
        bird_train_data = load_dataset(BIRD_TRAIN_JSON, DATASET_NAME, require_gold=True, split="train")
    print(f"Loaded {DATASET_NAME} dataset with {len(bird_train_data)} items.\n")
    
    schema_fingerprint = (schema_input_fingerprint(bird_train_data)
                          if args.dry_run and DATASET_NAME == "spider2_sqlite" and args.spider2_source
                          else prepare_schema_inputs(bird_train_data))
    print(f"Schema source: {TABLES_JSON or Path(SCHEMA_DIR).expanduser().resolve()}")
    print("Model CSV reuse checks IDs, coverage, execution results, and token records rather than fingerprints. Other caches remain fingerprint-validated.")
    source_fingerprint = dataset_fingerprint(bird_train_data)
    expected_ids = [question_key(row, i) for i, row in enumerate(bird_train_data)]
    if args.dry_run:
        print(json.dumps({"dataset_name": DATASET_NAME, "question_count": len(bird_train_data),
                          "database_count": len({row['db_id'] for row in bird_train_data}),
                          "source_dataset_fingerprint": source_fingerprint,
                          "output_dir": ARTIFACT_DIR, "model_results_dir": OUTPUT_DIR,
                          "first_question_id": expected_ids[0], "last_question_id": expected_ids[-1]}, indent=2))
        return 0
    Path(ARTIFACT_DIR).mkdir(parents=True, exist_ok=True)
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    # 2. Load or compute reference SQL results.
    gold_meta_path = Path(GOLD_CACHE_FILE + ".meta.json")
    gold_cache_valid = (os.path.exists(GOLD_CACHE_FILE) and gold_meta_path.is_file()
                        and json.loads(gold_meta_path.read_text(encoding="utf-8" )).get("fingerprint") == schema_fingerprint)
    if args.results_only:
        gold_results_cache = {}
    elif gold_cache_valid:
        print(f"[Cache found] Loading reference SQL results...")
        with open(GOLD_CACHE_FILE, 'rb') as f_cache:
            gold_results_cache = pickle.load(f_cache)
        print(f"Loaded {len(gold_results_cache)} reference results.\n")
    else:
        gold_results_cache = precompute_all_gold_results(bird_train_data)
    
    # 3. Run inference and evaluation for all models.
    model_csv_results = {}
    if args.results_only:
        for csv_path in sorted(Path(OUTPUT_DIR).glob("*_results.csv")):
            model_name = csv_path.name[:-len("_results.csv")]
            validate_existing_results(csv_path, expected_ids, DATASET_NAME, source_fingerprint)
            model_csv_results[model_name] = str(csv_path)
    for model_name, model_cfg in ([] if args.results_only else MODELS_TO_RUN.items()):
        csv_output_path = os.path.join(OUTPUT_DIR, f"{model_name}_results.csv")
        if result_csv_has_token_metrics(csv_output_path, schema_fingerprint, expected_ids):
            print(f"Results already exist for model {model_name}; skipping inference")
            model_csv_results[model_name] = csv_output_path
            continue
        try:
            if model_cfg["type"] == "local":
                csv_path = run_model_inference_and_eval(
                    model_name, model_cfg["model_path"], bird_train_data, gold_results_cache
                )
            elif model_cfg["type"] == "api":
                csv_path = run_api_inference_and_eval(
                    model_name, model_cfg, bird_train_data, gold_results_cache
                )
            model_csv_results[model_name] = csv_path
        except Exception as e:
            print(f"Model {model_name} failed: {str(e)}")

    if len(model_csv_results) < 2:
        raise RuntimeError("IRT calibration requires valid results from at least two models")
    
    # 4. Build the response matrix and calibrate 2PL-IRT.
    response_matrix, model_names = build_aligned_response_matrix(bird_train_data, model_csv_results)
    (
        irt_difficulties,
        irt_discriminations,
        valid_item_indices,
        irt_abilities,
        valid_student_indices,
    ) = run_irt_calibration_2pl(
        response_matrix, return_valid_student_indices=True
    )
    SAVE_DIR = os.path.join(ARTIFACT_DIR, "irt_2pl_calib_results")
    os.makedirs(SAVE_DIR, exist_ok=True)

    # 4.1. Save the original NumPy response matrix with shape [models, original items].
    np.save(os.path.join(SAVE_DIR, "response_matrix_raw.npy"), response_matrix)
    print(f"Original response matrix saved; shape: {response_matrix.shape}")

    # 4.2. Save model names in response-matrix row order.
    with open(os.path.join(SAVE_DIR, "model_names.json"), "w", encoding="utf-8") as f:
        json.dump(model_names, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(model_names)} model names")

    # 4.3. Save difficulty parameters in valid_item_indices order.
    np.save(os.path.join(SAVE_DIR, "irt_difficulties.npy"), irt_difficulties)
    print(f"Saved IRT difficulties for {len(irt_difficulties)} valid items")

    # 4.4. Save discrimination parameters in valid_item_indices order.
    np.save(os.path.join(SAVE_DIR, "irt_discriminations.npy"), irt_discriminations)
    print(f"Saved IRT discrimination parameters for {len(irt_discriminations)} valid items")

    # 4.5. Save retained real abilities only; synthetic abilities do not define absolute bounds.
    np.save(os.path.join(SAVE_DIR, "irt_abilities.npy"), irt_abilities)
    print(f"Saved ability parameters for {len(irt_abilities)} real respondents")

    # Align abilities with retained real respondent names.
    valid_model_names = [model_names[int(i)] for i in valid_student_indices]
    np.save(os.path.join(SAVE_DIR, "valid_student_indices.npy"), valid_student_indices)
    with open(os.path.join(SAVE_DIR, "model_names_valid.json"), "w", encoding="utf-8") as f:
        json.dump(valid_model_names, f, indent=2, ensure_ascii=False)

    # 4.6. Save valid item indices into the original training data for alignment.
    np.save(os.path.join(SAVE_DIR, "valid_item_indices.npy"), valid_item_indices)
    print(f"Saved valid item indices for training-data and difficulty alignment")

    # 4.7. Save the filtered real-response matrix aligned with abilities and item parameters.
    response_matrix_valid = response_matrix[np.ix_(valid_student_indices, valid_item_indices)]
    np.save(os.path.join(SAVE_DIR, "response_matrix_valid.npy"), response_matrix_valid)
    print(f"Filtered response matrix saved; shape: {response_matrix_valid.shape}")

    calibration_metadata = {
        "dataset_name": DATASET_NAME,
        "source_dataset_fingerprint": source_fingerprint,
        "method_version": "paper_irt_calibration_v4_group_normalized",
        "ability_scope": "retained_real_respondents_only",
        "ability_identification": "real_respondent_mean_zero_variance_one",
        "ability_reference_population": "retained_real_respondents",
        "ability_standardization_epsilon": float(
            IRT_CONFIG["2pl_ability_standardization_epsilon"]
        ),
        "raw_real_respondent_count": int(response_matrix.shape[0]),
        "retained_real_respondent_count": int(len(valid_student_indices)),
        "retained_item_count": int(len(valid_item_indices)),
        "virtual_respondents_used_for_item_calibration": int(
            IRT_CONFIG["vae_virtual_students"] + IRT_CONFIG["sample_virtual_students"]
        ),
        "real_group_total_weight_eta": float(
            IRT_CONFIG["2pl_real_group_weight_eta"]
        ),
        "response_weight_normalization": "sum_weights_equals_augmented_respondent_count",
        "item_initialization_correct_rate": "group_weighted",
        "regularization": {
            "theta": float(IRT_CONFIG["2pl_theta_regularization"]),
            "difficulty": float(IRT_CONFIG["2pl_difficulty_regularization"]),
            "log_discrimination": float(
                IRT_CONFIG["2pl_log_discrimination_regularization"]
            ),
        },
    }
    with open(os.path.join(SAVE_DIR, "calibration_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(calibration_metadata, f, indent=2, ensure_ascii=False)

    print("\nIRT calibration artifacts saved to:", os.path.abspath(SAVE_DIR))
    
    # 5. Save the dataset annotated with IRT difficulty.
    calibrated_dataset = []
    for out_idx, raw_item_idx in enumerate(valid_item_indices):
        item = bird_train_data[raw_item_idx]
        calibrated_item = {
            **item,
            "question_id": item.get("question_id", int(raw_item_idx)),
            "dataset_index": int(raw_item_idx),
            "source_dataset_fingerprint": source_fingerprint,
            "db_id": item["db_id"],
            "question": item["question"],
            "evidence": item.get("evidence", ""),
            "gold_sql": get_gold_sql(item, required=DATASET_NAME != "spider2_sqlite"),
            "human_difficulty": item.get("difficulty", "unknown"),
            "irt_difficulty": float(irt_difficulties[out_idx]),
            "irt_discrimination": float(irt_discriminations[out_idx])
        }
        calibrated_dataset.append(calibrated_item)
    
    with open(FINAL_IRT_JSON, 'w', encoding='utf-8') as f:
        json.dump(calibrated_dataset, f, indent=4, ensure_ascii=False)
    print(f"\nIRT annotations saved to: {FINAL_IRT_JSON}")
    
    # 6. Generate embeddings and train the pairwise difficulty ranker.
    print("\n" + "="*60)
    print(">>> Generating item embeddings and training the pairwise difficulty ranker")
    print("="*60)
    
    # Load cached embeddings.
    prepared_texts = [build_embedding_text(bird_train_data[idx], return_metadata=True) for idx in valid_item_indices]
    valid_texts = [text for text, prepared in prepared_texts]
    embedding_identity = {
        "dataset_name": DATASET_NAME,
        "ordered_question_ids": [question_key(row, i) for i, row in enumerate(calibrated_dataset)],
        "items_fingerprint": dataset_fingerprint(calibrated_dataset),
        "schema_source_format": "tables_json" if TABLES_JSON else "txt",
        "input_policy_version": EMBEDDING_INPUT_POLICY_VERSION,
        "embedding_token_budget": RANKER_CONFIG["embedding_token_budget"],
    }
    embedding_fingerprint = embedding_input_fingerprint(
        valid_texts, RANKER_CONFIG["embedding_model"]
    )
    embedding_metadata_file = EMBEDDING_CACHE_FILE + ".meta.json"
    embedding_cache_valid = False
    if os.path.exists(EMBEDDING_CACHE_FILE) and os.path.exists(embedding_metadata_file):
        with open(embedding_metadata_file, "r", encoding="utf-8") as f:
            embedding_metadata = json.load(f)
        embedding_cache_valid = (
            embedding_metadata.get("fingerprint") == embedding_fingerprint
            and embedding_metadata.get("embedding_model")
            == RANKER_CONFIG["embedding_model"]
            and embedding_metadata.get("item_count") == len(valid_texts)
            and all(embedding_metadata.get(key) == value for key, value in embedding_identity.items())
        )

    if embedding_cache_valid:
        with open(EMBEDDING_CACHE_FILE, 'rb') as f:
            embeddings = pickle.load(f)
        if len(embeddings) != len(valid_texts):
            raise ValueError("Training embedding cache length does not match the valid training item count")
        print(f"Loaded {len(embeddings)} embeddings from the validated cache")
    else:
        if os.path.exists(EMBEDDING_CACHE_FILE):
            print("Training embedding metadata is missing or inputs changed; regenerating embeddings")
        embeddings = get_batch_embeddings(valid_texts, model=RANKER_CONFIG["embedding_model"])
        with open(EMBEDDING_CACHE_FILE, 'wb') as f:
            pickle.dump(embeddings, f)
        with open(embedding_metadata_file, "w", encoding="utf-8") as f:
            json.dump(
                {
                    **embedding_identity,
                    "fingerprint": embedding_fingerprint,
                    "embedding_model": RANKER_CONFIG["embedding_model"],
                    "item_count": len(valid_texts),
                    "observable_input": "Concat(question,evidence,schema)",
                },
                f,
                indent=2,
                ensure_ascii=False,
            )
        print(f"Embeddings generated and cached at: {EMBEDDING_CACHE_FILE}")
    
    # Reuse the ranker only when embeddings, IRT difficulties, valid item order,
    # configuration, and method version match; otherwise retrain.
    ranker_training_contract = build_ranker_training_contract(
        embeddings,
        irt_difficulties,
        valid_item_indices=valid_item_indices,
    )
    ranker_training_contract.update({"dataset_name": DATASET_NAME,
                                     "items_fingerprint": embedding_identity["items_fingerprint"]})
    ranker_reusable, reuse_reason = ranker_checkpoint_is_reusable(
        RANKER_MODEL_PATH,
        ranker_training_contract,
    )
    if ranker_reusable:
        print(f"Compatible ranker found; skipping training: {RANKER_MODEL_PATH}")
        print(f"Reuse validation: {reuse_reason}")
        ranker_model = None
    else:
        print(f"Existing ranker cannot be reused; retraining: {reuse_reason}")
        ranker_model = train_ranker(
            embeddings,
            irt_difficulties,
            save_path=RANKER_MODEL_PATH,
            training_contract=ranker_training_contract,
        )
    
    print("\n" + "="*70)
    print(" Pipeline complete.")
    print(f"  - IRT difficulty annotations: {FINAL_IRT_JSON}")
    print(f"  - Difficulty ranker checkpoint: {RANKER_MODEL_PATH}")
    print("="*70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
