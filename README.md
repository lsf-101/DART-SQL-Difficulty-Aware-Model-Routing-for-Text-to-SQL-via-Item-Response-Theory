# DART-SQL

DART-SQL calibrates Text-to-SQL difficulty from historical execution outcomes, positions unseen questions with a pairwise ranker, and selects models using local execution accuracy, token efficiency, and estimation certainty. This repository provides calibration, routing, and direct SQL generation for BIRD.

Download the dataset and models separately, then run the pipeline to generate historical responses, checkpoints, routing decisions, and predictions. The generation code in this release implements direct generation only.

## Repository layout

```text
DART-SQL-release/
├── dart_sql/
│   ├── 2PL.py                     # Generate training responses, calibrate IRT, train the ranker
│   ├── test_2PL.py                # Position historical and unseen items using fixed anchors
│   ├── model_selector.py         # Retrieve candidates, weight evidence, and select models
│   ├── dataset_utils.py          # Dataset loading, identifiers, and schema handling
│   ├── embedding_input_budget.py # Embedding token budgets and schema truncation
│   └── run_direct_sql.py         # Fixed-model and routed direct SQL generation
├── configs/
│   ├── experiment.example.json   # Path placeholders and experiment settings
│   └── models.example.json       # Model aliases, weight-path placeholders, and devices
├── run_experiment.py             # Pipeline launcher
├── requirements.txt              # Base Python dependencies
├── requirements-gpu.txt          # Additional local vLLM environment
├── .gitignore                    # Upload allowlist
└── README.md
```

`test_2PL.py` is the unseen-item positioning stage, despite its filename. It is not a unit-test suite.

## Environment

Use Python 3.11. For a Linux shell, create and activate an environment before installing dependencies.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

For local model inference, install the GPU requirements in a Linux x86-64 environment with a compatible NVIDIA driver and sufficient GPU memory.

```bash
python -m pip install -r requirements-gpu.txt
python -m pip check
```

The base dependencies support ranker training from measured history and generation through an OpenAI-compatible server.

## Configuration


Use the same schema representation for calibration and positioning. For generation, schema TXT files are used when `data.dev_schema_dir` is set; otherwise the direct generator reads the SQLite schema. Keep these representations consistent with the intended experiment.

For each enabled model in `models.local.json`, replace `model_path` with its downloaded weight directory and set `cuda_visible_devices`. Set `"enabled": false` for models you will not run. Model aliases must match the generated history CSV names and routing selections. The calibration pipeline requires at least two valid solver response profiles, with enough nondegenerate items after filtering.

`training.history_gpu_ids` selects the shared GPU group for historical-response generation. For Dev generation, each model's `cuda_visible_devices` determines its devices. Set `llm_kwargs.tensor_parallel_size` when required by your hardware layout.

For a served model, replace its local-model entry with an OpenAI-compatible backend configuration.

```json
{
  "backend": "openai-compatible",
  "base_url": "your API base URL",
  "model": "your served model name",
  "api_key_env": "MODEL_API_KEY"
}
```

Set `MODEL_API_KEY` in the shell before using this entry. Embedding requests use `OPENAI_API_KEY` and optionally `OPENAI_BASE_URL`. The default embedding model is `text-embedding-3-large`; training and positioning must use the same embedding model and token-budget policy.

## Generate all results from source data

Leave `training.reuse_history` set to `false` on the first run. The calibration stage constructs historical response records, fits 2PL-IRT, and trains the pairwise ranker. Subsequent stages position items, select models, and generate Dev SQL.

```bash
python run_experiment.py all --config configs/experiment.local.json
```

The same pipeline can be run one stage at a time.

```bash
python run_experiment.py calibrate --config configs/experiment.local.json
python run_experiment.py position --config configs/experiment.local.json
python run_experiment.py route --config configs/experiment.local.json
python run_experiment.py generate --config configs/experiment.local.json
```

Before loading generation models, validate generation inputs with the following command after routing has completed.

```bash
python run_experiment.py generate --config configs/experiment.local.json --dry-run
```

Historical files are written as `<model_alias>_results.csv` beneath `history_dir`. They include `question_id`, `execution_result`, `prompt_tokens`, `completion_tokens`, and `total_tokens`. After generating them, set `training.reuse_history` to `true` to recalibrate from the measured responses. Existing complete valid CSVs are also reused by the inference stage, so use new history and output directories when changing experimental conditions.

To generate SQL with a fixed-model baseline, specify an enabled alias. This skips calibration, positioning, and routing.

```bash
python run_experiment.py all --config configs/experiment.local.json --model Model_8_Qwen3_5-27B
```

To rerun generation with a selection you have already produced, replace the quoted placeholder below with your generated `selected.json` path.

```bash
python run_experiment.py all --config configs/experiment.local.json --selection "your path"
```

## Generated files

For a full routed run, generated files appear beneath `work_dir` as follows.

```text
<work_dir>/
├── calibration/                 # IRT parameters, embedding caches, and ranker checkpoint
├── positions/                   # Historical and Dev scores and skeleton embeddings
├── selected.json                # Model routing decisions
├── routing_report.json          # Routing diagnostics
└── generation/routed/           # Predicted SQL, token usage, and generation records
```

