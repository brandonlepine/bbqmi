# Mechanistic interpretability study: bias (scaffold)

This repository is a lightweight, reproducible scaffold for mechanistic interpretability experiments focused on **bias** in language models (e.g., hidden-state probes, activation analysis, attribution, and group-wise comparisons).

## Directory overview

- **`configs/`**: Experiment configuration files (YAML).
- **`data/`**: Input datasets (raw) and derived datasets (processed).
  - **`data/raw/`**: Raw datasets you bring in (or generate).
  - **`data/processed/`**: Outputs from preprocessing scripts.
- **`scripts/`**: Runnable pipeline steps (CLI scripts).
- **`src/bbqmi/`**: Importable library code used by scripts.
- **`artifacts/`**: Model-related derived artifacts (e.g., cached activations).
- **`outputs/`**: Analysis outputs (tables/figures) intended for reporting.
- **`results/runs/<model_id>/<run_date>/`**: Model-scoped run outputs (analysis JSON, figures, activations, behavioral pilot).

## Data files included (example)

This scaffold includes a tiny example dataset you can replace with your real data:

- **`data/raw/example_bias_dataset.csv`**
  - **Columns**
    - **`id`**: Unique row identifier (string).
    - **`text`**: Input text/prompt to run through the model (string).
    - **`target`**: Optional target text/label for supervised tasks (string; may be empty).
    - **`group`**: Group membership for bias analysis (string; e.g., demographic group).
    - **`metadata_json`**: JSON-encoded string of additional metadata (string).

## Scripts (inputs/outputs)

- **`scripts/01_prepare_dataset.py`**
  - **Inputs**
    - `--input_csv`: Path to a CSV with at least `id,text,group` columns.
  - **Outputs**
    - Writes a processed CSV to `data/processed/` with a date suffix:
      - `dataset_processed_YYYY-MM-DD.csv`
    - Ensures required columns exist and normalizes basic types.

- **`scripts/02_extract_hidden_states.py`**
  - **Inputs**
    - `--input_csv`: Processed CSV (from step 01).
    - `--model_name`: Hugging Face model id (e.g. `gpt2`).
    - `--layer`: Which hidden-state layer to extract.
  - **Outputs**
    - Writes a compressed NumPy archive to `artifacts/` with a date suffix:
      - `hidden_states_YYYY-MM-DD.npz`
    - Contains arrays keyed by:
      - `hidden_states` (float32): shape `(n_examples, hidden_size)` pooled per example
      - `group` (str): group labels aligned to rows
      - `id` (str): ids aligned to rows

- **`scripts/03_probe_group_signal.py`**
  - **Inputs**
    - `--hidden_states_npz`: Output from step 02.
  - **Outputs**
    - Writes:
      - `outputs/probe_metrics_YYYY-MM-DD.json`
      - `outputs/probe_coefficients_YYYY-MM-DD.csv`

## Quickstart

Create an environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

Run the pipeline on the included example data:

```bash
python scripts/01_prepare_dataset.py --input_csv data/raw/example_bias_dataset.csv
python scripts/02_extract_hidden_states.py --input_csv data/processed/dataset_processed_$(date +%F).csv --model_name gpt2 --layer -1
python scripts/03_probe_group_signal.py --hidden_states_npz artifacts/hidden_states_$(date +%F).npz
```

## RunPod / H100 quickstart

This is the minimal sequence to run on a GPU instance (e.g., H100) without installing notebook/dev tooling.

- **Clone this repo**

```bash
git clone <YOUR_REPO_URL>
cd bbqmi
```

- **Create env + install minimal deps**

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.runpod.txt
pip install -e .
```

- **Install CUDA-enabled PyTorch**

PyTorch wheels are CUDA-version-specific. On most CUDA 12.x images, this works (adjust if your image differs):

```bash
pip install --index-url https://download.pytorch.org/whl/cu124 torch
```

- **Get the BBQ dataset**

This repo expects the BBQ dataset repo at `data/BBQ/`:

```bash
git clone https://github.com/nyu-mll/BBQ.git data/BBQ
```

- **Prepare stimuli (writes dated outputs into `data/processed/`)**

```bash
python scripts/prepare_stimuli.py
```

- **(Optional) Download models from Hugging Face**

```bash
python scripts/download_models.py --help
```

- **Run the behavioral pilot (writes into a model-scoped run dir)**

```bash
python scripts/behavioral_pilot.py \
  --device cuda \
  --model_path /workspace/bbqmi/models/llama2-13b \
  --model_id llama2-13b
```

- **Extract activations (writes into the same run dir)**

```bash
python scripts/extract_activations.py \
  --device cuda \
  --model_path /workspace/bbqmi/models/llama2-13b \
  --model_id llama2-13b \
  --stimuli stimuli_so.json \
  --output_subdir so
```

- **One-command end-to-end run (recommended)**

This runs: `prepare_stimuli.py` → `behavioral_pilot.py` → `extract_activations.py` → analyses (and optional GI).

```bash
python scripts/run_pipeline.py \
  --device cuda \
  --model_path /workspace/bbqmi/models/llama2-13b \
  --model_id llama2-13b \
  --include_gi
```

- **Run the decomposition/ablation analysis (reads/writes in the same run dir)**

```bash
python scripts/analyze_decomposition.py \
  --ablation_only \
  --device cuda \
  --model_path /workspace/bbqmi/models/llama2-13b \
  --model_id llama2-13b \
  --alpha 14.0
```

## Notes / conventions

- Output file names created by scripts include the run date suffix `YYYY-MM-DD`.
- `data/`, `artifacts/`, and `outputs/` contents are ignored by git (folders are tracked).

## Run directories (model-scoped outputs)

Most BBQ scripts now read/write inside a **model-scoped run directory**:

- **Default run dir**: `results/runs/<model_id>/<run_date>/`
- **Subdirectories**:
  - `activations/so/` and `activations/gi/` (plus `manifest.json`)
  - `behavioral_pilot/`
  - `analysis/`
  - `figures/`

Scripts accept:

- **`--model_id`**: used to separate outputs across models (defaults to `model_path.name`)
- **`--run_date`**: defaults to today; if that folder doesn’t exist yet, scripts fall back to the newest existing run for that `model_id`
- **`--run_dir`**: explicit override (useful for resuming a specific run)

## Multi-model runs (example)

```bash
for mid in llama2-13b llama2-7b; do
  python scripts/behavioral_pilot.py --device cuda --model_path "/workspace/bbqmi/models/$mid" --model_id "$mid"
  python scripts/extract_activations.py --device cuda --model_path "/workspace/bbqmi/models/$mid" --model_id "$mid" --stimuli stimuli_so.json --output_subdir so
  python scripts/analyze_decomposition.py --ablation_only --device cuda --model_path "/workspace/bbqmi/models/$mid" --model_id "$mid" --alpha 14.0
done
```
