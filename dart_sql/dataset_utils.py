"""Shared BIRD/Spider identities, Gold SQL and schema representation.

No gold query or SQL AST is ever used to construct an embedding or prompt.
Missing question IDs are assigned before any filtering of the source array.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

RANKER_METHOD_VERSION = "paper_pairwise_ranker_v6_sqrt_gap_weight"
SCHEMA_FILE_PATTERNS = ("{db_id}.txt", "{db_id}_schema.txt")


def canonical_id(value):
    if value is None or isinstance(value, (bool, float)):
        raise ValueError(f"Invalid question_id: {value!r}")
    text = str(value).strip()
    if not text:
        raise ValueError("Empty question_id")
    return str(int(text)) if re.fullmatch(r"[+-]?\d+", text) else text


def question_key(item, index):
    return canonical_id(item.get("question_id", index))


def get_gold_sql(item, required=True):
    # Spider's `sql` is an AST dictionary; its executable query is `query`.
    for key in ("SQL", "query", "gold_sql", "sql"):
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    if required:
        raise ValueError("Missing executable Gold SQL: expected SQL/query/gold_sql or string sql")
    return ""


def normalize_dataset(payload, dataset_name="bird", require_gold=False, split=None):
    if dataset_name not in ("bird", "spider", "spider2_sqlite"):
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    if not isinstance(payload, list) or not payload:
        raise ValueError("Dataset must be a nonempty JSON array")
    result, seen = [], set()
    for index, raw in enumerate(payload):
        if not isinstance(raw, dict):
            raise ValueError(f"Dataset row {index} must be an object")
        if raw.get("dataset_name", dataset_name) != dataset_name:
            raise ValueError(f"Dataset row {index} belongs to {raw.get('dataset_name')}, expected {dataset_name}")
        row = dict(raw)
        identifier = question_key(row, index)
        if identifier in seen:
            raise ValueError(f"Duplicate question_id: {identifier}")
        seen.add(identifier)
        for key in ("db_id", "question"):
            if not isinstance(row.get(key), str) or not row[key].strip():
                raise ValueError(f"Question {identifier}: missing {key}")
        if row.get("evidence") is not None and not isinstance(row["evidence"], str):
            raise ValueError(f"Question {identifier}: evidence must be text or null")
        row.setdefault("question_id", index)
        row.setdefault("dataset_index", index)
        row["dataset_name"] = dataset_name
        if split is not None:
            if row.get("dataset_split", split) != split:
                raise ValueError(f"Question {identifier}: dataset split mismatch")
            row["dataset_split"] = split
        if require_gold:
            if dataset_name == "spider2_sqlite":
                from spider2_sqlite_data import validate_reference
                validate_reference(row.get("spider2_reference", {}))
            else:
                get_gold_sql(row)
        result.append(row)
    return result


def load_dataset(path, dataset_name="bird", require_gold=False, split=None):
    return normalize_dataset(json.loads(Path(path).read_text(encoding="utf-8-sig")),
                             dataset_name, require_gold, split)


def dataset_fingerprint(items):
    """Stable across IRT enrichment; includes original identities and Gold SQL."""
    records = [dict(question_id=question_key(item, index), db_id=item["db_id"],
                    question=item["question"], evidence=item.get("evidence") or "",
                    gold_sql=get_gold_sql(item, required=False),
                    dataset_name=item.get("dataset_name", "bird"))
               for index, item in enumerate(items)]
    for record, item in zip(records, items):
        if item.get("dataset_name") == "spider2_sqlite":
            from spider2_sqlite_data import reference_digest
            record["reference_digest"] = reference_digest(item.get("spider2_reference", {}))
    return hashlib.sha256(json.dumps(records, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":")).encode("utf-8")).hexdigest()


def read_schema_text(schema_dir, db_id, file_pattern="auto"):
    if (not isinstance(db_id, str) or not db_id.strip()
            or db_id in (".", "..") or any(char in db_id for char in '/\\:\x00')):
        raise ValueError(f"Invalid db_id: {db_id!r}")
    root = Path(schema_dir).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Schema directory does not exist: {root}")
    if file_pattern not in ("auto", *SCHEMA_FILE_PATTERNS):
        raise ValueError(f"Unsupported schema file pattern: {file_pattern}")
    patterns = SCHEMA_FILE_PATTERNS if file_pattern == "auto" else (file_pattern,)
    paths = [(root / pattern.format(db_id=db_id)).resolve() for pattern in patterns]
    if any(not path.is_relative_to(root) for path in paths):
        raise ValueError(f"Schema path escapes directory: {db_id}")
    matches = [path for path in paths if path.is_file()]
    if len(matches) != 1:
        if not matches:
            raise FileNotFoundError(f"Missing schema TXT for {db_id}; tried {paths}")
        raise ValueError(f"Ambiguous schema TXT for {db_id}; set --schema-file-pattern")
    text = matches[0].read_text(encoding="utf-8-sig").strip()
    if not text:
        raise ValueError(f"Empty schema TXT for {db_id}: {matches[0]}")
    return text


def render_schema_info(info):
    if isinstance(info, str):
        return info
    tables = info.get("table_names_original") or info.get("table_names")
    columns = info.get("column_names_original") or info.get("column_names")
    if not tables or not columns:
        raise ValueError("Schema entry needs table_names[_original] and column_names[_original]")
    return "\n".join(
        f"Table `{name}` (Columns: {', '.join(col[1] for col in columns if col[0] == index)})"
        for index, name in enumerate(tables))


def load_schema_map(items, schema_dir=None, tables_path=None, file_pattern="auto"):
    if (schema_dir is None) == (tables_path is None):
        raise ValueError("Supply exactly one of --schema-dir or --tables-json")
    databases = sorted({item["db_id"] for item in items})
    if schema_dir is not None:
        return {db_id: read_schema_text(schema_dir, db_id, file_pattern) for db_id in databases}
    payload = json.loads(Path(tables_path).read_text(encoding="utf-8-sig"))
    if not isinstance(payload, list):
        raise ValueError("tables.json must be an array")
    entries = {}
    for info in payload:
        db_id = info["db_id"]
        if db_id in entries:
            raise ValueError(f"Duplicate tables.json db_id: {db_id}")
        entries[db_id] = info
    missing = set(databases) - entries.keys()
    if missing:
        raise ValueError(f"Missing tables.json schemas: {sorted(missing)}")
    return {db_id: render_schema_info(entries[db_id]) for db_id in databases}


def resolve_database_path(db_root, db_id):
    root = Path(db_root).expanduser().resolve()
    if not isinstance(db_id, str) or not db_id or db_id in (".", "..") or any(c in db_id for c in '/\\:\x00'):
        raise ValueError(f"Invalid database ID: {db_id}")
    nested = root / db_id / f"{db_id}.sqlite"
    flat = root / f"{db_id}.sqlite"
    path = (nested if nested.is_file() else flat).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise FileNotFoundError(f"Missing database: tried {nested} and {flat}")
    return path


def database_signature(items, db_root):
    root = Path(db_root).expanduser().resolve()
    databases = []
    for db_id in sorted({item["db_id"] for item in items}):
        path = resolve_database_path(root, db_id)
        if not path.is_relative_to(root) or not path.is_file():
            raise FileNotFoundError(f"Missing database or invalid db_id: {path}")
        stat = path.stat()
        databases.append((str(path), stat.st_size, stat.st_mtime_ns))
    return databases


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
