# Pipeline Documentation — `pipeline_modified.py`

A comprehensive, statement-by-statement reference for the AAPL stock price forecasting pipeline.
The pipeline trains and evaluates multiple deep learning models across **two scenarios**
(`"With Sentiment"` / `"Without Sentiment"`) and **three prediction horizons** (1, 8, 16 days ahead),
iterated over **three lookback windows** (96, 32, and 16 days), for a total of 18 experimental
configurations per full run.

---

## Table of Contents

1. [Imports](#1-imports)
2. [MLflow & DagsHub Setup](#2-mlflow--dagshub-setup)
3. [Global Configuration](#3-global-configuration)
4. [Checkpoint System — `is_completed` & `mark_completed`](#4-checkpoint-system)
5. [Logging Utilities — `cprint` & `ColorLog`](#5-logging-utilities)
6. [Reproducibility — `set_seed`](#6-reproducibility--set_seed)
7. [TCN Building Blocks — `Chomp1d` & `TemporalBlock`](#7-tcn-building-blocks)
8. [Normalization Modules](#8-normalization-modules)
   - [8.1 RevIN](#81-revin)
   - [8.2 SANorm](#82-sanorm)
   - [8.3 FANorm](#83-fanorm)
9. [PatchTST Pretraining — `pretrain_patchtst_model`](#9-patchtst-pretraining)
10. [Excel Output Helpers](#10-excel-output-helpers)
11. [Data Loading & Preprocessing — `load_and_split_data`](#11-data-loading--preprocessing)
12. [Embedding Modules](#12-embedding-modules)
    - [12.1 PositionalEmbedding](#121-positionalembedding)
    - [12.2 TimeFeatureEmbedding](#122-timefeatureembedding)
    - [12.3 DataEmbedding](#123-dataembedding)
13. [Attention — `ProbSparseAttention`](#13-attention--probsparseattention)
14. [Encoder Architecture](#14-encoder-architecture)
    - [14.1 DistillingLayer](#141-distillinglayer)
    - [14.2 EncoderLayer](#142-encoderlayer)
    - [14.3 InformerEncoder](#143-informerencoder)
15. [Decoder — `DecoderLayer`](#15-decoder--decoderlayer)
16. [Dataset Classes](#16-dataset-classes)
    - [16.1 InformerDataset](#161-informerdataset)
    - [16.2 PatchTSTDataset](#162-patchtst-dataset)
17. [Loss Function — `MultiFactorLoss`](#17-loss-function--multifactorloss)
18. [Model Architectures](#18-model-architectures)
    - [18.1 TransformerEncoderLayerLayerNorm](#181-transformerencoderlayerlayernorm)
    - [18.2 PatchTST](#182-patchtst)
    - [18.3 Informer](#183-informer)
    - [18.4 LSTMModel](#184-lstmmodel)
    - [18.5 TCNModel](#185-tcnmodel)
    - [18.6 EnsembleModel](#186-ensemblemodel)
19. [Training Infrastructure](#19-training-infrastructure)
    - [19.1 EarlyStopping](#191-earlystopping)
    - [19.2 train_torch_model](#192-train_torch_model)
20. [Metric & Evaluation Helpers](#20-metric--evaluation-helpers)
21. [Pipeline Orchestration Functions](#21-pipeline-orchestration-functions)
22. [Ablation Suite — Model Name Reference](#22-ablation-suite--model-name-reference)
23. [Data Flow Diagram](#23-data-flow-diagram)

---

## 1. Imports

```python
import pandas as pd
import numpy as np
```

`pandas` handles all tabular data operations: loading the CSV, merging stock price data with news sentiment by date, grouping and aggregating, and writing results to Excel files. `numpy` provides the fast array operations used throughout — slicing lookback windows, reshaping prediction tensors, computing metrics, and stacking MC-Dropout samples.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
```

The core PyTorch stack. `torch` provides tensor arithmetic and GPU/CPU device management. `nn` is the module system — every layer (Linear, LSTM, Conv1d), every loss function, and every full model is a subclass of `nn.Module`. `F` provides stateless functional operations such as `F.pad` (used in SANorm to pad sequences before slicing) and `F.interpolate` (used in PatchTST to adjust patch count). `Dataset` is the abstract base class for all three custom dataset objects (`InformerDataset`, `PatchTSTDataset`), and `DataLoader` wraps a dataset into an iterable that yields shuffled or sequential mini-batches.

```python
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
```

`MinMaxScaler` maps each feature to `[0, 1]` via:

$$x_{\text{scaled}} = \frac{x - x_{\min}}{x_{\max} - x_{\min}}$$

Two independent scalers are maintained throughout the pipeline — one for the Close price target (`scaler_target`) and one for all covariates (`scaler_cov`). Keeping them separate is critical because `scaler_target` is used to inverse-transform model predictions back to real dollar values, and mixing it with other features would corrupt that calculation.

```python
from torch.nn.utils import weight_norm
from sklearn.mixture import GaussianMixture
from IPython.display import display, HTML
```

`weight_norm` reparameterises any `nn.Module`'s weight matrix `W` into a direction vector `v` and a scalar magnitude `g`, such that:

$$W = g \cdot \frac{v}{\|v\|}$$

This decouples learning the *scale* of the weights from their *direction*, which stabilises training in the TCN layers by preventing gradient interference between the two aspects of the weight. `GaussianMixture` fits a 3-component probabilistic model to the historical volatility signal to label market regimes. `display(HTML(...))` renders formatted coloured HTML output inside Jupyter/Kaggle notebooks.

```python
import math, os, warnings, optuna, random, copy, sys, re, gc
import matplotlib.pyplot as plt
import dagshub, mlflow
```

- `math` — used for constants and functions in positional encoding (`math.log(10000.0)`) and ProbSparse complexity (`math.log(L_Q)`)
- `os` — `os.makedirs`, `os.path.exists`, `os.environ` for file management and credential injection
- `warnings.filterwarnings('ignore')` — suppresses sklearn/PyTorch deprecation warnings to keep logs readable
- `optuna` — Bayesian hyperparameter optimisation; manages the search space and trial pruning
- `random` — Python's built-in RNG; seeded for reproducibility
- `copy.deepcopy` — creates a fully independent clone of a trained model for TTA, ensuring the original weights are not modified
- `re` — `re.sub('<[^<]+>', '', msg)` strips HTML tags inside `cprint` so that the terminal sees clean text
- `gc.collect()` — triggers Python's garbage collector after each lookback window loop to free tensors and avoid OOM
- `mlflow` — experiment tracking; logs parameters, metrics, artifacts, and model weights to DagsHub
- `dagshub` — provides the hosted MLflow tracking server

> **Note:** `from captum import ...` has been removed. The XAI (Explainable AI) module based on Captum that previously computed saliency maps and feature attributions is no longer present in this version. The pipeline is now purely focused on forecasting accuracy and uncertainty quantification.

---

## 2. MLflow & DagsHub Setup

```python
dagshub_token = "941879c4b4456ada2c76193a8f16aa79c955460b"
os.environ["MLFLOW_TRACKING_USERNAME"] = "tinyboi2021"
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token
os.environ["MLFLOW_TRACKING_URI"] = "https://dagshub.com/tinyboi2021/vc-stock-prediction-new.mlflow"
```

Sets three environment variables that the MLflow HTTP client reads automatically. DagsHub uses HTTP Basic Authentication — the username and token are transmitted as credentials on every API request. The URI tells the client where to send experiment data. The token is embedded directly in the script rather than being fetched from Kaggle's secrets vault (as in previous versions), making this version portable to any local Python environment.

```python
mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
```

`set_tracking_uri` points the active MLflow client at the DagsHub server. `set_experiment` selects (or creates) the experiment namespace under which all runs are grouped, using the easily-configurable global variable `MLFLOW_EXPERIMENT_NAME`. Every subsequent `mlflow.start_run()` call will appear as a child run under this experiment in the DagsHub UI.

```python
torch.set_float32_matmul_precision('medium')
```

On NVIDIA Ampere-class GPUs (A100, RTX 3090, H100), this enables TF32 (TensorFloat-32) precision for matrix multiplications. TF32 uses a 10-bit mantissa instead of the standard 23-bit float32 mantissa — this gives approximately 8× faster matrix multiplication with negligible numerical difference for deep learning. Has no effect on CPU or pre-Ampere GPUs.

```python
optuna.logging.set_verbosity(optuna.logging.INFO)
```

Sets Optuna's own internal logger to INFO level, so trial numbers, objective values, and elapsed times are printed but internal debug messages are suppressed. This gives useful progress feedback during the 100-trial hyperparameter search without flooding the log.

```python
try:
    mlflow.enable_system_metrics_logging()
except Exception:
    pass
```

Attempts to enable automatic logging of CPU usage, RAM, and GPU metrics to MLflow. This feature requires `psutil` (and optionally `pynvml` for GPU) to be installed. The `try/except` silently skips it if those packages are absent, ensuring the pipeline does not crash in minimal environments.

---

## 3. Global Configuration

### Multi-Run Mode

```python
TARGET_SCENARIO = ["With Sentiment", "Without Sentiment"]
TARGET_HORIZON  = [1, 8, 16]
RUNNER_NAME     = "Hareesh K S"
```

A key design change: both `TARGET_SCENARIO` and `TARGET_HORIZON` are now **lists**. The `main()` function loops over all list elements, so a single execution trains models for all 6 scenario × horizon combinations (2 × 3). Each team member changes `RUNNER_NAME` before running — this string is attached as an MLflow tag to every run, enabling per-person filtering in the DagsHub leaderboard.

The team assignment comments above these lines document the intended split:
- Paul S → Horizons 1 (both scenarios)
- Tom T → Horizons 8 (both scenarios)
- Hareesh K S → Horizons 16 (both scenarios)

### Paths and Storage

```python
DATA_PATH      = "data/aapl_stock_sentiment_new_dataset_2017_2026.csv"
DATASET_NAME   = os.path.basename(DATA_PATH)
MODEL_SAVE_DIR = "./Saved_Models"
PRED_SAVE_DIR  = "./Prediction_Excel_Sheets"
```

`DATA_PATH` points to the merged AAPL stock + sentiment CSV. Unlike the Kaggle version that used `/kaggle/input/`, this is a relative path for local execution. Both `MODEL_SAVE_DIR` and `PRED_SAVE_DIR` are created immediately with `os.makedirs(..., exist_ok=True)` so the script does not crash if they don't exist.

The output Excel file path is no longer a global variable — it is computed dynamically inside the `main()` evaluation loops per scenario and per horizon:
```python
current_output_file = f".Results/Results_{scenario.replace(' ', '_')}_H{pred_len}.xlsx"
```
This dynamic generation ensures that results from different horizons naturally segment into their own Excel sheets rather than crashing on list-to-string conversion errors or overwriting each other. It is passed down through `process_standalone_models()` and `evaluate_ensembles()` to ensure unified loop-safe behavior.

### Training Budget Constants

```python
EPOCHS    = 300
PATIENCE  = 20
N_TRIALS  = 100
```

`EPOCHS = 300` is the hard maximum. In practice, early stopping (patience 20) terminates training much sooner — typically between 40 and 120 epochs. `PATIENCE = 20` means the model is allowed 20 consecutive epochs without validation MSE improvement before training halts. `N_TRIALS = 100` sets the Optuna hyperparameter search budget per model type; with Bayesian TPE sampling, 100 trials reliably identifies near-optimal hyperparameter combinations.

### Feature Flags

```python
ENABLE_REVIN       = True
ENABLE_CUSTOM_LOSS = True
```

These flags toggle two major architectural choices without code changes:
- `ENABLE_REVIN = False` → RevIN's `_normalize` method returns the input unchanged (with dummy zero-mean, unit-stdev statistics). The model trains on raw scaled values.
- `ENABLE_CUSTOM_LOSS = False` → `train_torch_model` always uses `nn.MSELoss`, regardless of scenario. This allows a direct comparison of the custom multi-factor loss's contribution to accuracy.

```python
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

Detects whether a CUDA-capable GPU is available and sets the compute device accordingly. All models and batch tensors are moved to `DEVICE` before training.

---

## 4. Checkpoint System

### `is_completed(scenario, lookback, horizon, model_name)`

This function prevents expensive re-training of models that have already finished. Rather than reading a local CSV (the old approach), it queries **MLflow on DagsHub** directly:

```python
query = (
    f"params.Model = '{model_name}' and "
    f"params.Scenario = '{scenario}' and "
    f"params.Horizon = '{horizon}' and "
    f"params.Lookback = '{lookback}' and "
    f"params.Dataset = '{DATASET_NAME}' and "
    f"tags.Runner = '{RUNNER_NAME}' and "
    f"status = 'FINISHED'"
)
exp = mlflow.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)
runs = mlflow.search_runs(experiment_ids=[exp.experiment_id], filter_string=query)
```

Six fields must all match: model name, scenario, horizon, lookback, dataset name, and the current runner's name. A run with `status = 'FINISHED'` means the `with mlflow.start_run():` block completed cleanly without raising an exception — MLflow sets this automatically.

**Why this is better than a local CSV checkpoint:**
The CSV approach has a race condition: if two team members run the same configuration simultaneously, both see an empty CSV and both begin training, wasting resources. The MLflow approach is server-side and atomic. It also survives machine restarts, notebook session resets, and environment changes, because the data lives on the remote DagsHub server.

If the DagsHub API call fails (network error, expired token), the function defaults to `return False`, causing the model to re-run rather than being silently skipped. Failing open is safer than failing closed here.

### `mark_completed(scenario, lookback, horizon, model_name)`

```python
def mark_completed(scenario, lookback, horizon, model_name):
    """Deprecated: No longer needed."""
    pass
```

This is now a no-op. MLflow sets `status = 'FINISHED'` automatically when a `with mlflow.start_run():` block exits cleanly. The function exists only to avoid `AttributeError` at the many existing call sites in `process_standalone_models` and `evaluate_ensembles`.

---

## 5. Logging Utilities

### Environment Detection

```python
def _is_jupyter():
    shell = get_ipython().__class__.__name__
    return shell in ('ZMQInteractiveShell', 'TerminalInteractiveShell')

_IN_JUPYTER = _is_jupyter()
_IN_KAGGLE  = "KAGGLE_KERNEL_RUN_TYPE" in os.environ
```

`get_ipython()` is only defined inside IPython environments. `ZMQInteractiveShell` is the Jupyter notebook kernel; `TerminalInteractiveShell` is IPython at the command line. The `try/except NameError` handles the case where the code runs as a plain Python script. `_IN_KAGGLE` checks for an environment variable that Kaggle's runner infrastructure sets automatically.

### ANSI Terminal Codes

```python
if _IN_KAGGLE:
    _R = _B = _CYAN = _GREEN = _YELLOW = _MAGENTA = ""
else:
    _R      = "\033[0m"   # Reset
    _B      = "\033[1m"   # Bold
    _CYAN   = "\033[96m"  # Bright cyan
    _GREEN  = "\033[92m"  # Bright green
    _YELLOW = "\033[93m"  # Bright yellow
    _MAGENTA= "\033[95m"  # Bright magenta
```

ANSI escape codes are byte sequences that most terminals interpret as formatting instructions. The format is `\033[` (ESC character) + `code` + `m`. Kaggle's output cell does not render ANSI codes — they appear as raw text — so all codes are set to empty strings when running on Kaggle, preventing garbled output.

### `cprint(msg)`

```python
def cprint(msg):
    clean = re.sub('<[^<]+>', '', msg).replace('&nbsp;', ' ')
    print(clean)
    if _IN_JUPYTER:
        display(HTML(msg))
```

`re.sub('<[^<]+>', '', msg)` applies a regular expression that matches anything starting with `<`, containing one or more non-`<` characters, and ending with `>` — the standard pattern for HTML tags. This strips all tags so the terminal receives clean plain text. If inside a Jupyter environment, the original `msg` (with HTML tags intact) is rendered as formatted output via `display(HTML(...))`. This dual-output approach gives rich coloured output in notebooks while remaining readable in plain terminals.

### `ColorLog`

```python
class ColorLog:
    HEADER = OKBLUE = OKCYAN = OKGREEN = WARNING = FAIL = ENDC = BOLD = UNDERLINE = ''
```

These class-level attributes are legacy HTML format tokens kept as empty strings. Many `cprint(f"...{ColorLog.OKGREEN}...{ColorLog.ENDC}...")` calls exist throughout the pipeline; since these are empty strings now, they produce no visible output but prevent `AttributeError`. The actual coloured output comes from the ANSI codes used inside the static methods:

| Method | Output format | Typical use |
|--------|--------------|-------------|
| `info(msg)` | `[INFO]` in bold cyan | Status updates, checkpoint checks |
| `success(msg)` | `[OK]` in bold green | Metric results, saved models |
| `warn(msg)` | `[WARN]` in bold yellow | Non-fatal errors, NaN detection |
| `section(msg)` | `--- msg ---` in bold magenta | Section dividers |
| `extreme(msg)` | 60-char `=` border in bold green | Phase starts, major milestones |

`extreme()` prints a 60-character border of `=` signs above and below the uppercased message, making critical milestones visually prominent in thousands of lines of training output.

---

## 6. Reproducibility — `set_seed`

```python
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True)
```

Achieving fully deterministic deep learning requires seeding every independent source of randomness:

- **`random.seed(42)`** — seeds Python's built-in RNG, which Optuna's TPE sampler and any Python-level `random.choice` calls depend on
- **`np.random.seed(42)`** — seeds NumPy's global RNG, controlling `np.random.choice` in mock data generation and any NumPy sampling operations
- **`torch.manual_seed(42)`** — seeds PyTorch's CPU RNG, which controls weight initialisation (e.g., `nn.Linear` uses Kaiming uniform), dropout masks on CPU, and `DataLoader` worker seeds
- **`torch.cuda.manual_seed(42)` / `manual_seed_all(42)`** — seeds the GPU RNG for the current device and all GPU devices; GPU operations without this produce different results across runs even with all CPU seeds fixed
- **`cudnn.deterministic = True`** — forces cuDNN to use deterministic algorithm implementations rather than the fastest non-deterministic ones. Some cuDNN algorithms use atomic operations that can produce floating-point rounding differences across runs
- **`cudnn.benchmark = False`** — disables cuDNN's auto-tuner, which selects algorithm variants on the first forward pass. The selection process itself is non-deterministic
- **`CUBLAS_WORKSPACE_CONFIG = ":4096:8"`** — allocates a fixed-size workspace buffer for cuBLAS deterministic kernels. Required by `torch.use_deterministic_algorithms(True)` for certain operations like `bmm` and `gemm` on CUDA
- **`torch.use_deterministic_algorithms(True)`** — raises a `RuntimeError` if any PyTorch operation attempts to use a non-deterministic implementation, enforcing strict reproducibility at the cost of some performance

`set_seed(42)` is called at the start of every Optuna objective function, ensuring each hyperparameter trial starts from the same random state rather than inheriting the state from the previous trial.

---

## 7. TCN Building Blocks

### Why Temporal Convolutional Networks?

A TCN replaces the sequential recurrence of LSTMs with **causal dilated convolutions**. Key advantages:
- **Causality** — output at time $t$ depends only on inputs at times $\leq t$ (no future leakage)
- **Long-range dependencies** — dilation exponentially expands the receptive field with depth, matching LSTM's capability at far lower computational cost
- **Full parallelism** — unlike RNNs, all timesteps are processed simultaneously during training (no sequential dependency)
- **Stable gradients** — residual connections allow gradient flow across many layers without vanishing/exploding

### `Chomp1d`

```python
def forward(self, x):
    return x[:, :, :-self.chomp_size].contiguous()
```

Causality in a `Conv1d` layer requires that zero-padding is applied **only to the left** (past) side of the sequence. PyTorch's `padding` argument pads symmetrically on both sides, so after a convolution with `padding=p`, the last `p` output elements contain information derived from the right-side (future) padding — violating causality.

`Chomp1d` removes these future-contaminated elements by slicing off the last `chomp_size` positions:

```
Input after symmetric padding:  [0  0  x₁  x₂  x₃  x₄  0  0]
After Conv1d (kernel=3):         [y₁  y₂  y₃  y₄  y₅  y₆]
After Chomp (chomp_size=2):      [y₁  y₂  y₃  y₄]
```

The result is that `y_t` only sees `x_{t-k}` for `k ≥ 0`. `.contiguous()` ensures the sliced tensor occupies a contiguous block of memory, which is required by subsequent operations.

### `TemporalBlock`

```python
self.net = nn.Sequential(conv1, chomp1, relu1, dropout1,
                          conv2, chomp2, relu2, dropout2)
self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
```

One residual block applies two causal dilated convolutions in sequence:

$$\text{output} = \text{ReLU}\bigl(\text{Net}(x) + \text{Downsample}(x)\bigr)$$

The **dilated convolution** equation is:

$$y[t] = \sum_{k=0}^{K-1} w[k] \cdot x[t - d \cdot k]$$

where $d$ is the dilation factor and $K$ is the kernel size. With dilation $d = 2^i$ for layer $i$, the effective receptive field grows exponentially:

$$\text{Receptive Field} = 1 + 2(K-1)\sum_{i=0}^{n-1} 2^i = 1 + 2(K-1)(2^n - 1)$$

For `kernel_size=2` and 6 layers: receptive field = $1 + 2 \times 1 \times 63 = 127$ timesteps — comfortably covering any of the three lookback windows (16, 32, 96).

If the number of input and output channels differ, a 1×1 convolution (`downsample`) projects the residual to the right channel count before adding. This is the standard residual connection trick from ResNet.

**Weight norm** — each convolutional layer is wrapped with `weight_norm`. The weight matrix `W` is reparameterised as:

$$W = g \cdot \frac{v}{\|v\|}$$

The gradient with respect to `v` is always orthogonal to `v` itself, preventing the weight vector from growing unconstrained and improving convergence speed by approximately decoupling the optimisation of magnitude and direction.

---

## 8. Normalization Modules

All three normalisation classes implement the same interface:

```python
module.forward(x, mode='norm' | 'denorm',
               mean=None, stdev=None, gamma=None, beta=None, target_idx=None)
```

When `mode='norm'`, the module normalises `x` and returns `(x_norm, mean, stdev, gamma, beta)`. The statistics are stored and later passed back to `mode='denorm'` to reverse the transformation on the model's predictions.

The `target_idx=0` argument in the denormalisation call restricts the inverse transform to only the Close price column (column 0), preventing statistics from other feature channels from contaminating the final price prediction.

---

### 8.1 RevIN

**Source:** Kim et al., ICLR 2022 — *"Reversible Instance Normalization for Accurate Time-Series Forecasting against Distribution Shift"*
[https://openreview.net/forum?id=cGDAkQo1C0p](https://openreview.net/forum?id=cGDAkQo1C0p)

#### Motivation

A model trained on AAPL at \$150 per share may generalise poorly when the stock reaches \$200 because it has memorised the *absolute price level* rather than *relative patterns* (trends, cycles, volatility structure). RevIN forces every input window to zero mean and unit variance, making the model distribution-agnostic.

#### Forward Normalisation

Given input $\mathbf{x} \in \mathbb{R}^{B \times L \times C}$ (batch × time × features):

**Step 1 — Compute instance statistics across the time axis:**

$$\mu = \frac{1}{L}\sum_{t=1}^{L} x_t \quad \in \mathbb{R}^{B \times 1 \times C}$$

$$\sigma^2 = \frac{1}{L}\sum_{t=1}^{L}(x_t - \mu)^2 \quad \in \mathbb{R}^{B \times 1 \times C}$$

Both `mean` and `stdev` are computed with `.detach()` — gradients are **not** propagated through the statistics. This is intentional: we do not want the model to learn to manipulate its own normalisation statistics as a shortcut during backpropagation.

The variance is clamped: `torch.clamp(var, min=1e-5)` prevents division by near-zero in constant sequences (e.g., a trading halt where price does not move for several days).

**Step 2 — Normalise:**

$$\hat{x}_t = \frac{x_t - \mu}{\sigma + \varepsilon}$$

**Step 3 — Learnable affine rescaling (when `affine=True`):**

$$\tilde{x}_t = \hat{x}_t \cdot \gamma + \beta$$

where $\gamma \in \mathbb{R}^C$ (initialised to **1**) and $\beta \in \mathbb{R}^C$ (initialised to **0**) are learnable parameters registered with `nn.Parameter`. These allow the model to learn, for example, that volatility features should be slightly amplified relative to price features.

#### Reverse Denormalisation

After the model produces a prediction $\hat{y}$:

**Step 1 — Reverse affine (if `target_idx` is set, only for the Close price column):**

$$\hat{y}_{\text{raw}} = \frac{\hat{y} - \beta[\text{target\_idx}]}{|\gamma[\text{target\_idx}]| + \varepsilon}$$

Note: $|\gamma|$ is used instead of $\gamma$ to prevent division by a negative value, which would flip the sign of predictions.

**Step 2 — Restore original scale:**

$$\hat{y}_{\text{real}} = \hat{y}_{\text{raw}} \cdot \sigma[\text{target\_idx}] + \mu[\text{target\_idx}]$$

#### `enable_revin=False` Pass-through

When `enable_revin=False`, `_normalize` returns the input unchanged with dummy statistics (zero mean, unit stdev, `None` gamma/beta), and `_denormalize` is a no-op that returns its input. This allows complete ablation of normalisation without any changes to the model forward pass or training loop.

---

### 8.2 SANorm

**Source:** Liu et al., NeurIPS 2023 — *"Adaptive Normalization for Non-stationary Time Series Forecasting: A Temporal Slice Perspective"*
[https://openreview.net/forum?id=5BqDSw8r5j](https://openreview.net/forum?id=5BqDSw8r5j)

#### Motivation

RevIN applies a **single global mean and variance** across the entire input window of length $L$. For non-stationary financial series, this is suboptimal: a 96-day lookback window might contain a calm month followed by an earnings-driven shock. The global mean is dominated by the shock, biasing the normalisation for the calm period and vice versa.

SANorm addresses this by dividing the window into $S = 4$ equal-length temporal slices and normalising each independently. This allows the model to adapt to locally different statistical regimes within a single input sequence.

#### Slice Decomposition

Given $\mathbf{x} \in \mathbb{R}^{B \times L \times C}$:

**Step 1 — Pad if needed:**
```python
pad_len = (num_slices - L % num_slices) % num_slices
x_padded = F.pad(x, (0, 0, 0, pad_len), mode='replicate')
```
If $L$ is not divisible by $S$, the sequence is extended by replicating the last timestep. `mode='replicate'` is chosen over zero-padding so that the padded region has realistic statistics rather than a zero baseline.

**Step 2 — Reshape into slices:**
$$\mathbf{x}_{\text{sliced}} = \text{reshape}(\mathbf{x}_{\text{padded}}) \in \mathbb{R}^{B \times S \times (L/S) \times C}$$

**Step 3 — Per-slice statistics:**
$$\mu_s = \frac{1}{L/S}\sum_{t \in \text{slice}_s} x_{s,t} \quad \in \mathbb{R}^{B \times S \times 1 \times C}$$

$$\sigma_s = \sqrt{\text{Var}(x_{\text{slice}_s}) + \varepsilon} \quad \in \mathbb{R}^{B \times S \times 1 \times C}$$

$$\hat{x}_{s,t} = \frac{x_{s,t} - \mu_s}{\sigma_s}$$

#### MLP-Based Adaptive Affine Transform

Unlike RevIN's fixed learnable $\gamma, \beta$, SANorm uses a small MLP to predict data-conditioned affine parameters from the per-slice mean:

```python
self.affine_net = nn.Sequential(
    nn.Linear(C, C // 2),
    nn.ReLU(),
    nn.Linear(C // 2, 2*C)   # outputs [gamma, beta] concatenated
)
```

$$[\gamma_s, \beta_s] = \text{MLP}(\mu_s) \quad \in \mathbb{R}^{B \times S \times 1 \times C}$$

$$\tilde{x}_{s,t} = \hat{x}_{s,t} \cdot \gamma_s + \beta_s$$

The MLP input is the slice mean $\mu_s$, making the affine parameters **conditioned on local statistical context**. In a high-volatility slice, the MLP might learn to produce a larger $\gamma$ to preserve more of the signal's spread; in a low-volatility slice, it might contract it. This adaptivity is SANorm's key advantage over RevIN.

#### Denormalisation with Last-Slice Statistics

Denormalisation uses only the statistics from the **last slice** ($s = S-1$):

```python
last_mean  = slice_mean[:, -1, :].detach()
last_stdev = slice_stdev[:, -1, :].detach()
```

The last slice represents the most recent sub-series and therefore best approximates the statistical regime the model is forecasting into. Using global statistics (as RevIN does) would over-weight earlier historical segments.

$$\hat{y}_{\text{raw}} = \frac{\hat{y} - \beta_{S-1}}{\gamma_{S-1} + \varepsilon}, \quad \hat{y}_{\text{real}} = \hat{y}_{\text{raw}} \cdot \sigma_{S-1} + \mu_{S-1}$$

**Key difference from RevIN:** RevIN applies one global learnable affine; SANorm applies $S$ locally-adaptive data-driven affine transforms — one per temporal segment.

---

### 8.3 FANorm

**Source:** *"Frequency Adaptive Normalization for Non-stationary Time Series Forecasting"*, arXiv 2409.20371

#### Motivation

Both RevIN and SANorm operate entirely in the **time domain**. FANorm works in the **frequency domain**, decomposing the input into a low-frequency trend component and a high-frequency seasonal component via the Fast Fourier Transform (FFT), then applying separate learnable affine transforms to each. This reflects the established financial time-series principle that price trends and seasonal patterns have fundamentally different statistical structures and should not be normalised together.

#### Step 1 — Global Standardisation

$$\mu = \frac{1}{L}\sum_{t=1}^{L} x_t \in \mathbb{R}^{B \times 1 \times C}, \quad \sigma = \sqrt{\text{Var}(x) + \varepsilon} \in \mathbb{R}^{B \times 1 \times C}$$

$$x_{\text{std}} = \frac{x - \mu}{\sigma + \varepsilon}$$

Zero-centring before FFT is essential: a large non-zero mean would dominate the DC (bin 0) component of the spectrum, making it impossible to cleanly separate trend from seasonal. The mean and stdev are stored (detached) for later denormalisation.

#### Step 2 — Frequency Decomposition

```python
x_freq = torch.fft.rfft(x.permute(0, 2, 1), dim=-1)  # [B, C, L//2+1]
```

`rfft` (Real FFT) exploits the Hermitian symmetry of the spectrum for real-valued signals, producing only the positive-frequency half: $L/2 + 1$ complex-valued frequency bins. Each bin $f$ represents a sinusoidal component that completes $f$ full cycles over the sequence length $L$.

**Learnable frequency cutoff:**

$$\alpha = \sigma(\theta) \in (0, 1), \quad k = \max\left(1, \lfloor\alpha \cdot (L/2 + 1)\rfloor\right)$$

where $\theta = \texttt{log\_freq\_ratio}$ is a scalar learnable parameter. The sigmoid mapping $\sigma(\cdot)$ ensures $\alpha$ stays in $(0, 1)$. The cutoff $k$ is the number of low-frequency bins allocated to the trend component.

**Initial value:** $\theta = -1.95$ because $\sigma(-1.95) \approx 0.125$, placing the initial cutoff at the bottom 12.5% of the spectrum — keeping approximately the lowest 12 bins (out of ~97 bins for $L=192$) as trend. This is a reasonable prior for stock data, where the trend component is typically much more dominant than high-frequency noise.

During training, backpropagation adjusts $\theta$ to find the cutoff that minimises the forecasting loss. The gradient flows through the `sigmoid`, through the `int()` approximation (gradient is treated as 1 via the straight-through estimator since `int()` has zero gradient almost everywhere), and into $\theta$.

**Separation:**

$$X_{\text{trend}}[f] = \begin{cases} X_{\text{freq}}[f] & f < k \\ 0 & f \geq k \end{cases}, \quad X_{\text{season}}[f] = \begin{cases} 0 & f < k \\ X_{\text{freq}}[f] & f \geq k \end{cases}$$

$$x_{\text{trend}} = \text{IRFFT}(X_{\text{trend}}), \quad x_{\text{season}} = \text{IRFFT}(X_{\text{season}})$$

The two signals are recovered via the inverse RFFT. By construction: $x_{\text{trend}} + x_{\text{season}} = x_{\text{std}}$ (perfect reconstruction).

#### Step 3 — Independent Learnable Affine Transforms

```python
out = trend * self.trend_gamma   + self.trend_beta   \
    + seasonal * self.season_gamma + self.season_beta
```

Four learnable parameter vectors (each $\in \mathbb{R}^C$) independently scale and shift the trend and seasonal components before combining them:

$$\text{out} = x_{\text{trend}} \cdot \gamma_T + \beta_T + x_{\text{season}} \cdot \gamma_S + \beta_S$$

The model can learn, for example, to heavily weight the trend component ($\gamma_T \gg \gamma_S$) for long-horizon forecasting, or to amplify the seasonal component for short-horizon predictions where weekly patterns dominate.

#### Denormalisation

Only the global standardisation is reversed — the frequency decomposition is not inverted:

$$\hat{y}_{\text{real}} = \hat{y} \cdot \sigma[\text{target\_idx}] + \mu[\text{target\_idx}]$$

The model is trained end-to-end to produce predictions that only need the global scale restoration. The frequency-decomposed normalised space becomes the model's natural operating domain.

---

## 9. PatchTST Pretraining

```python
def pretrain_patchtst_model(model, train_loader, val_loader, epochs, lr, patience, model_save_path):
```

### What is Masked Patch Reconstruction (MPR)?

MPR is a **self-supervised pretraining** strategy inspired by BERT's masked language modelling. The intuition: randomly zero out 40% of the input patches, then train the model to reconstruct the original masked patch values from the remaining visible context. This forces the model to learn rich, generalisable representations of time-series structure before it ever sees a price prediction label.

After pretraining, the `pretrain_head` is discarded and the encoder weights are used as the initialisation for the supervised forecasting task (`PatchTST_Pretrained` variant). This is expected to converge faster and generalise better than random initialisation.

### Loss Computation

```python
rec_patches, patches_orig, mask = model(enc, pretrain=True, mask_ratio=0.4)
mask_expanded = mask.unsqueeze(-1).expand_as(rec_patches)
loss_matrix   = criterion(rec_patches, patches_orig)        # per-element MSE
loss = (loss_matrix * mask_expanded).sum() / (mask_expanded.sum() + 1e-8)
```

`criterion = nn.MSELoss(reduction='none')` returns the element-wise squared errors. The `mask_expanded` selects only the errors at **masked positions** — not the visible patches that the model can trivially copy. Dividing by `mask_expanded.sum()` normalises by the actual count of masked elements, which varies stochastically from batch to batch (Bernoulli sampling with $p = 0.4$). The $10^{-8}$ prevents division by zero in the edge case of an entirely unmasked batch.

### Gradient Norm Logging

```python
grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
if mlflow.active_run():
    mlflow.log_metric("grad_norm", grad_norm.item(), step=epoch)
```

`clip_grad_norm_` clips all parameter gradients so that the global L2 norm does not exceed 1.0, then **returns the norm before clipping**. Logging the pre-clip norm to MLflow is a useful diagnostic: a norm that grows exponentially signals numerical instability; a norm that shrinks toward zero signals vanishing gradients. Note the check `if mlflow.active_run()` — this prevents an error if pretraining is called outside an active MLflow run context.

### Best Model Saving

```python
if avg_train_loss < best_loss:
    best_loss = avg_train_loss
    torch.save(model.state_dict(), model_save_path)
...
model.load_state_dict(torch.load(model_save_path))
```

Only the `state_dict` (the tensor values of all parameters and buffers) is saved, not the full model object. This is more portable — the model architecture is re-instantiated from code, and only the learned weights are read from disk. At the end of pretraining, the best epoch's weights are reloaded.

---

## 10. Excel Output Helpers

### `save_prediction_details`

Exports per-model actual vs. predicted prices to Excel for manual inspection and downstream analysis:

```python
actuals_real = scaler_target.inverse_transform(actuals_flat).reshape(N, H)
preds_real   = scaler_target.inverse_transform(preds_flat).reshape(N, H)
```

Predictions arrive as MinMax-scaled values in $[0, 1]$. `inverse_transform` reverses the scaling:

$$x_{\text{real}} = x_{\text{scaled}} \cdot (x_{\max} - x_{\min}) + x_{\min}$$

For each horizon step $h$, three columns are written: `Actual_Step_h`, `Predicted_Step_h`, `Error_Step_h`. Each model gets its own worksheet (sheet name capped at 31 characters — Excel's hard limit). If the file already exists, the sheet is replaced (not the full file) via `if_sheet_exists='replace'`, allowing multiple models to be written to the same workbook sequentially.

### `save_results_to_excel`

```python
os.makedirs(os.path.dirname(filename), exist_ok=True)
```

This explicit directory creation was added in this version to fix a `FileNotFoundError` that occurred when the `.Results/` directory did not yet exist when the first result was being saved. If the file already exists, the new results are **concatenated** to the existing DataFrame and the file is overwritten — this accumulates results from all models within a scenario/horizon run.

---

## 11. Data Loading & Preprocessing — `load_and_split_data`

### Dataset Description

The input is a CSV covering AAPL (Apple Inc.) stock data from 2017 to 2026. Each row represents one trading day and contains:
- **Close** — daily closing price (the prediction target)
- **RSI** — Relative Strength Index (momentum oscillator)
- **SMA** — Simple Moving Average
- **ADX** — Average Directional Index (trend strength)
- **Gold** — Gold spot price (macro risk proxy)
- **USDJPY** — USD/JPY exchange rate (macro risk proxy)
- **sentiment_score** — Daily average sentiment score derived from news articles via LLM-based NLP

### Mock Data Fallback

```python
if not os.path.exists(DATA_PATH):
    # Generate 2200 synthetic daily observations
    'Close': np.random.randn(2200).cumsum() + 100,  # Random walk from $100
    'ADX':   np.random.uniform(0, 50, 2200),
    ...
```

If the CSV is missing, the function generates 2,200 synthetic trading days with realistic statistical properties (random walk for price, uniform distributions for bounded indicators). This allows the full pipeline structure to be tested without the real dataset.

### Data Merging and Multi-Row Handling

When the CSV has multiple rows per date (e.g., one row per news article), the function separates technical indicators from sentiment:

```python
df_grouped    = df_raw.groupby('Date').agg(sentiment_score=('sentiment_score', 'mean')).reset_index()
df_technicals = df_raw.drop(columns=['sentiment_score']).groupby('Date').first().reset_index()
df_full = pd.merge(df_technicals, df_grouped, on='Date', how='left')
```

Technical indicators are deduplicated by taking the first row per date (OHLCV data is identical across rows for the same date). Sentiment is averaged across all articles published on that date. A left join preserves all trading days, filling sentiment-absent days with 0.

### Sentiment Lag — The Most Critical Data Integrity Step

```python
df_full['sentiment_score'] = df_full['sentiment_score'].shift(1).fillna(0)
```

This single line prevents **look-ahead bias** — the most common and most dangerous mistake in financial ML. Today's news articles are typically published during or after market hours. Using today's sentiment to predict today's closing price means the model has access to information that would not be available when a real forecast must be made. Shifting by one day ensures the model sees only *yesterday's* sentiment when predicting *today's* price. The first row's sentiment is set to 0 (neutral) since there is no prior day's data.

### Static 60/20/20 Chronological Split

```python
val_start_idx  = int(n * 0.60)   # Validation starts at 60%
test_start_idx = int(n * 0.70)   # Test starts at 70%
```

The dataset is split chronologically — **never shuffled**. For $n = 2200$ trading days:
- Training: rows 0–1319 (1,320 days, ~2017–2021)
- Validation: rows 1320–1539 (220 days, ~2021–2022)
- Test: rows 1540–2199 (660 days, ~2022–2026)

The split is 60/10/30 (not 60/20/20 as the docstring says — the 10% validation set is small but sufficient for hyperparameter selection given the large training set).

### Overlap Windows — Ensuring Full Lookback Context

```python
val_slice  = slice(max(0, val_start_idx - max_lookback), test_start_idx)
test_slice = slice(max(0, test_start_idx - max_lookback), None)
val_overlap_len  = val_start_idx - val_slice.start
test_overlap_len = test_start_idx - test_slice.start
```

Each split is extended backward by `max_lookback = 96` rows. This ensures the **first prediction window** at the start of each split has a complete 96-step encoder context. Without this, the first 96 windows at each boundary could not be formed — effectively wasting data. `val_overlap_len` and `test_overlap_len` record how many leading rows are "warm-up" context from the previous split; evaluation functions skip predictions from these warm-up rows.

### GMM Market Regime Detection

The pipeline identifies three market regimes: low, medium, and high volatility. A Gaussian Mixture Model clusters the 20-day rolling return volatility signal.

**Step 1 — Log-returns:**

$$r_t = \log\!\left(\frac{P_t}{P_{t-1}}\right) = \log(P_t + \varepsilon) - \log(P_{t-1} + \varepsilon)$$

Log-returns are used instead of simple returns because they are more symmetric around zero, more stationary, and more normally distributed — all desirable for GMM fitting.

**Step 2 — Rolling standard deviation:**

$$\sigma_t^{\text{roll}} = \text{std}(r_{t-19}, \ldots, r_t)$$

A 20-day rolling window (approximately one trading month) smooths the return series into a volatility signal.

**Step 3 — GMM fitting (train data only):**

$$p(\sigma) = \sum_{k=1}^{3} \pi_k \cdot \mathcal{N}(\sigma \mid \mu_k, \sigma_k^2)$$

**Crucially**, the GMM is fitted only on `volatility[:train_end]`. Fitting on validation or test data would expose training to future volatility patterns — a form of look-ahead bias in regime detection. After fitting, `gmm.predict(volatility)` assigns a regime label $\{0, 1, 2\}$ to every row based on which Gaussian component has the highest posterior probability.

### Missing Value Imputation — No-Leakage Strategy

```python
df_full[numeric_cols] = df_full[numeric_cols].ffill()          # 1. Forward-fill all splits
df_full.iloc[:train_slice.stop][numeric_cols] = \
    df_full.iloc[:train_slice.stop][numeric_cols].bfill()       # 2. Backward-fill training only
df_full[numeric_cols] = df_full[numeric_cols].fillna(0)        # 3. Zero-fill remaining
```

Three imputation passes in order:
1. **Forward-fill** — propagates the last valid value forward in time. Safe everywhere because past information flows forward naturally.
2. **Backward-fill training only** — fills NaNs at the very start of the training set (e.g., RSI is undefined for the first 14 days). Restricted to training data because allowing backward-fill into validation/test would let future data influence imputation.
3. **Zero-fill** — catches columns that are entirely empty (rare edge case).

### Two-Scaler Strategy

```python
scaler_cov    = MinMaxScaler()  # Fitted on training covariates only
scaler_target = MinMaxScaler()  # Fitted on training Close price only
```

Separate scalers allow independent inverse-transformation of predictions back to real dollar values. Both are **fitted on training data only** (`.fit_transform`) and then **applied** to validation and test (`.transform`). This prevents test set statistics from influencing the scaling boundaries.

### Sentiment Exception — Bypassing MinMax Scaling

```python
cov_chunk[:, cov_sent_idx] = raw_chunk[:, sent_col_idx]
```

After scaling all covariates with `scaler_cov`, the raw (unscaled) sentiment scores are restored. Why? Sentiment is a signed value in $[-1, +1]$ from the LLM. MinMax scaling to $[0, 1]$ destroys the sign information — a score of $-0.8$ (strongly negative news) and $+0.8$ (strongly positive news) would become $0.1$ and $0.9$, making the direction-of-sentiment ambiguous. More importantly, `MultiFactorLoss` applies thresholds like `sentiment < -0.05` to assign loss weights; if sentiment were MinMax-scaled, those thresholds would be wrong.

---

## 12. Embedding Modules

### 12.1 `PositionalEmbedding`

Implements the sinusoidal positional encoding from *"Attention Is All You Need"* (Vaswani et al., 2017):

$$PE(\text{pos}, 2i) = \sin\!\left(\frac{\text{pos}}{10000^{2i/d_{\text{model}}}}\right), \quad PE(\text{pos}, 2i+1) = \cos\!\left(\frac{\text{pos}}{10000^{2i/d_{\text{model}}}}\right)$$

Even dimensions receive sine encoding; odd dimensions receive cosine encoding. The denominator $10000^{2i/d}$ creates a geometric progression of frequencies — low dimensions oscillate slowly (long-range position), high dimensions oscillate quickly (local position). Together, every position gets a unique fingerprint across all $d_{\text{model}}$ dimensions.

**Key property:** The encoding of position $\text{pos} + k$ can be expressed as a linear function of the encoding at $\text{pos}$, allowing the model to generalise to sequence lengths it has never seen during training.

The encoding matrix is pre-computed once in `__init__` up to `max_len = 5000` positions and stored with `register_buffer`. This keeps it on the correct device automatically when the model is moved, while excluding it from the optimiser's parameter updates (it is fixed, not learned).

### 12.2 `TimeFeatureEmbedding`

```python
self.embed_month   = nn.Embedding(13, d_model)  # Months 1-12 + padding index 0
self.embed_day     = nn.Embedding(32, d_model)  # Days 1-31 + padding
self.embed_weekday = nn.Embedding(7,  d_model)  # Mon(0) through Sun(6)
self.embed_hour    = nn.Embedding(25, d_model)  # Hours 0-23 + padding
```

`nn.Embedding(N, d)` is a learnable lookup table of $N$ dense $d$-dimensional vectors. Given an integer index, it returns the corresponding row vector. Unlike one-hot encoding, the learned embeddings allow the model to represent semantic similarity (e.g., Friday embeddings may become similar to Thursday embeddings because both are pre-weekend trading days).

The four embeddings are **summed** (not concatenated) to maintain the fixed `d_model` dimension:

$$\text{temporal} = \text{embed\_month}(m) + \text{embed\_day}(d) + \text{embed\_weekday}(w) + \text{embed\_hour}(h)$$

`torch.nan_to_num(x, nan=0.0).long()` handles edge cases where date parsing fails and produces NaN values, replacing them with index 0 before the integer conversion.

### 12.3 `DataEmbedding`

Combines three signals into a single `d_model`-dimensional token per timestep:

```python
self.value_embedding = nn.Conv1d(c_in, d_model, kernel_size=3, padding=1,
                                  padding_mode='circular')
```

The value embedding uses a 1D convolution with `kernel_size=3`, so each timestep's token incorporates information from its immediate neighbours (one step before and after). `padding_mode='circular'` wraps the sequence ends around each other rather than zero-padding, which is more natural for time series — the "edge" of a trading window is not meaningfully different from the interior.

After the convolution, the tensor is transposed from `[B, d_model, L]` back to `[B, L, d_model]` to match the Transformer's expected format.

**Final token:**

$$\text{token}_t = \text{ValueEmbed}(x_t) + \text{PosEmbed}(t) + \text{TimeEmbed}(\text{mark}_t)$$

Both positional and temporal embeddings are **optional** (controlled by `use_positional` and `use_temporal` flags). The ablation variants `Informer_NoTemporal`, `Informer_NoPositional`, and `Informer_NoBothEmbeds` test whether removing each embedding type hurts accuracy — this directly answers the research question of how much calendar and position information contribute to forecasting performance.

---

## 13. Attention — `ProbSparseAttention`

### Standard Attention Complexity

In standard scaled dot-product attention:

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right) V$$

The attention matrix $QK^\top \in \mathbb{R}^{L_Q \times L_K}$ has $O(L^2)$ elements. For $L = 96$: 9,216 values — manageable. For $L = 512$: 262,144 values — 28× more expensive. ProbSparse reduces this to $O(L \log L)$.

### The ProbSparse Insight

Empirically, most queries are "lazy" — their attention distributions are nearly uniform and contribute little discriminative information. Only a small fraction of queries are "active" — they concentrate attention on a few dominant keys. ProbSparse identifies and computes full attention only for the active queries.

### Algorithm Walkthrough

**Step 1 — Sample a subset of keys:**
```python
u        = int(factor * math.log(L_Q))   # Sample size ~5·ln(96) ≈ 22
rand_idx = torch.randint(0, L_K, (B, n_heads, u))
K_sample = torch.gather(K, 2, idx_expanded)
```
Instead of computing $Q \cdot K^\top$ in full, each query is first scored against a random sample of $u = c \cdot \ln(L_Q)$ keys. This reduces the importance estimation from $O(L^2)$ to $O(L \log L)$.

**Step 2 — Sparsity score:**

$$\bar{M}(q_i, K) = \max_j\!\left(\frac{q_i k_j^\top}{\sqrt{d_k}}\right) - \frac{1}{u}\sum_j\!\left(\frac{q_i k_j^\top}{\sqrt{d_k}}\right)$$

The difference between the maximum score and the mean score measures how far the query's attention deviates from uniform. A query with high $\bar{M}$ attends selectively to specific keys ("active"). A query with low $\bar{M}$ attends approximately equally to all keys ("lazy").

**Step 3 — Select top-$u$ queries:**
```python
_, top_u_idx = torch.topk(M_score, u, dim=-1)
Q_selected   = torch.gather(Q, 2, top_u_idx_expanded)
```
Only the $u$ queries with the highest sparsity scores get full attention computed against all $L_K$ keys.

**Step 4 — Assign lazy queries via causal cumulative mean:**
```python
cum_v      = V_valid.cumsum(dim=2)
valid_counts = valid_mask.float().cumsum(dim=2).clamp(min=1.0)
context    = (cum_v / valid_counts).clone()
```
For the remaining "lazy" queries, instead of spending compute on their (nearly uniform) attention, a causal cumulative mean of the value vectors is assigned. Position $t$ gets the mean of $V_0, V_1, \ldots, V_t$ — not the global mean, which would violate causality.

**Step 5 — Insert active query outputs:**
```python
context.scatter_(2, top_u_idx_expanded, context_selected)
```
`scatter_` writes the computed outputs for active queries back into their original positions in the cumulative-mean context tensor.

**Step 6 — Output projection:**
```python
context = context.transpose(1, 2).contiguous().view(B, L_Q, -1)
return self.out_proj(context)
```
Multi-head outputs are concatenated and linearly projected back to `d_model`.

### Complexity Comparison

| Method | Time | Memory |
|--------|------|--------|
| Standard Attention | $O(L^2)$ | $O(L^2)$ |
| ProbSparse Attention | $O(L \log L)$ | $O(L \log L)$ |

For $L = 96$: standard = 9,216 ops; ProbSparse ≈ 22 × 96 ≈ 2,112 ops — approximately 4× fewer operations on this sequence length, with the savings growing for longer sequences.

---

## 14. Encoder Architecture

### 14.1 `DistillingLayer`

Applied between encoder layers (except the last) to progressively halve the sequence length, creating a memory hierarchy:

```
x [B, L, d] → permute → Conv1d(3, pad=1) → permute → LayerNorm → ELU → permute → MaxPool1d(stride=2) → permute → [B, L/2, d]
```

**ELU vs ReLU:** The distilling layer uses ELU (Exponential Linear Unit):

$$\text{ELU}(x) = \begin{cases} x & x > 0 \\ \alpha(e^x - 1) & x \leq 0 \end{cases}$$

Unlike ReLU, ELU produces smooth gradients for negative inputs (gradient = $\alpha e^x > 0$), preventing the "dying ReLU" problem where neurons permanently output zero and their gradients become permanently zero.

**MaxPool with stride 2** selects the maximum in each 3-element window, halving the sequence. This is a principled downsampling: taking the maximum preserves the most prominent activations at each location, rather than averaging (which can wash out peaks).

After $n$ distilling layers, the sequence is $L / 2^n$ steps long. Three encoder stacks with 3, 2, and 1 layers respectively process sequences of length $L$, $L/2$, and $L/4$.

### 14.2 `EncoderLayer`

A standard Transformer encoder block with Post-LN ordering:

$$x' = \text{LayerNorm}(x + \text{Dropout}(\text{Attention}(x, x, x)))$$
$$x'' = \text{LayerNorm}(x' + \text{Dropout}(\text{FFN}(x')))$$

The FFN uses **GELU activation**:

$$\text{GELU}(x) = x \cdot \Phi(x) \approx x \cdot \sigma(1.702x)$$

where $\Phi$ is the CDF of the standard normal distribution. GELU is smoother than ReLU and stochastically similar to applying dropout followed by linear transformation, which regularises the representation implicitly. It has been empirically superior in Transformer architectures since GPT-1.

The FFN has a 2× expansion: `Linear(d → 2d) → GELU → Linear(2d → d)`. This bottleneck structure (expand then compress) is standard and appears to help Transformers learn richer intermediate representations.

The `prob_sparse` flag switches between `ProbSparseAttention` and PyTorch's `nn.MultiheadAttention` (full attention). The `Informer_FullAttn` ablation uses `prob_sparse=False` to test whether ProbSparse is actually beneficial for this sequence length.

### 14.3 `InformerEncoder`

Implements a multi-resolution **encoder pyramid**:

```python
for i, stack in enumerate(self.stacks):
    inp_len = x.shape[1] // (2**i)
    inp     = x[:, -inp_len:, :]   # Most recent inp_len steps
    out     = stack(inp)
    stack_outs.append(out)
```

With `e_layers = [3, 2, 1]`:
- **Stack 0** (3 layers + 2 distilling): sees the full sequence $L$, produces output of length $\lfloor L/4 \rfloor$
- **Stack 1** (2 layers + 1 distilling): sees the most recent $L/2$ steps, produces output of length $\lfloor L/4 \rfloor$
- **Stack 2** (1 layer): sees the most recent $L/4$ steps, produces output of the same length

All stack outputs are aligned to the shortest length and concatenated:

```python
min_len      = min([s.shape[1] for s in stack_outs])
aligned_outs = [s[:, -min_len:, :] for s in stack_outs]
out          = torch.cat(aligned_outs, dim=1)
```

The concatenated output gives the decoder access to representations at three different temporal scales simultaneously — coarse (long-range context from Stack 0), medium (Stack 1), and fine (Stack 2, most recent). This is the Informer's key architectural contribution.

---

## 15. Decoder — `DecoderLayer`

```python
def forward(self, x, cross, tgt_mask=None, cross_attn=True):
    # 1. Masked self-attention on the decoder sequence
    new_x = self.self_attn(x, x, x, attn_mask=tgt_mask)
    x = self.norm1(x + self.dropout(new_x))
    
    # 2. Cross-attention: decoder queries the encoder output
    if cross_attn:
        new_x, _ = self.cross_attn(x, cross, cross)
        x = self.norm2(x + self.dropout(new_x))
    
    # 3. Feed-forward
    new_x = self.ff(x)
    x = self.norm_final(x + self.dropout(new_x))
    return x
```

**Self-attention with causal mask:**

$$\text{tgt\_mask}[i, j] = \begin{cases} \text{True (mask out)} & j > i \\ \text{False} & j \leq i \end{cases}$$

This upper-triangular boolean mask, when passed to attention, causes positions $j > i$ to receive logits of $-10^4$ — effectively 0 weight after softmax. This enforces that position $i$ can only attend to positions $0, 1, \ldots, i$ in the decoder, maintaining the auto-regressive property.

**Cross-attention** always uses full `nn.MultiheadAttention` (never ProbSparse). The decoder positions serve as queries ($Q = x$) and the encoder output provides both keys and values ($K = V = \text{enc\_out}$). This is the mechanism by which the decoder "reads" the compressed multi-resolution representation from the encoder.

**Cross-attention bypass (`cross_attn=False`):** The `Informer_DecOnly` ablation has no encoder. The `dec_only=True` flag causes `cross_attn=False` to be passed to each `DecoderLayer.forward`, skipping the cross-attention sub-layer entirely. This tests whether the encoder is actually necessary for this forecasting task.

---

## 16. Dataset Classes

### 16.1 `InformerDataset`

Used for all Informer variants. Implements the encoder-decoder windowing scheme from the Informer paper.

**Window layout:**
```
s_begin ─────────────── s_end
[      encoder input (seq_len)     ]
              r_begin ──────────────────── r_end
              [  label (seq_len//2)  ][  prediction (pred_len)  ]
                                       ↑
                                    f_start (masking boundary)
```

- `seq_x = data[s_begin:s_end]` — full encoder input
- `seq_y = data[r_begin:r_end]` — decoder context + future ground truth
- `seq_y_input` — the **masked** decoder input (what the model sees at inference)

**Masking logic — strict real-world simulation:**

```python
seq_y_input[f_start:, :] = 0.0                           # Zero out all future features
seq_y_input[f_start:, target_col_idx] = last_close        # Warm-start: last known Close
seq_y_input[f_start, sent_col_idx]    = tomorrow_sentiment # T+1 sentiment is available
```

All prediction-horizon features are zeroed. The Close price in the horizon is set to the last known Close (persistence warm-start for the decoder). Only tomorrow's sentiment is unmasked because today's news articles are published before tomorrow's market opens — this information genuinely *is* available at forecast time.

**Return tuple (9 elements):**
`(seq_x, seq_x_mark, seq_y_input, seq_y, seq_y_mark, target, target_sentiment, target_adx, target_regime)`

`target_regime` is the GMM regime label at the last encoder timestep — used by DRO training to split losses by market condition.

### 16.2 `PatchTST` Dataset

Simpler encoder-only format for PatchTST, LSTM, and TCN:

```python
seq_x  = data[s_begin:s_end]                             # Encoder input window
target = data[s_end:s_end + pred_len, 0:1]               # Future Close prices only
```

Sentiment and ADX for the prediction horizon are extracted to supply `MultiFactorLoss` during training. If absent (`sent_col_idx is None`), zero arrays are returned.

**Return tuple (5 elements):**
`(seq_x, target, target_sentiment, target_adx, target_regime)`

The `__len__` method returns `max(0, len(data) - seq_len - pred_len + 1)` — the `max(0, ...)` guard prevents a negative value if the dataset is shorter than one window (which can happen during testing with very small datasets).

---

## 17. Loss Function — `MultiFactorLoss`

Standard MSE treats all forecast errors equally regardless of market context. `MultiFactorLoss` encodes three domain-informed beliefs:

1. **Errors during negative news are more costly** — negative sentiment causes faster, larger price moves
2. **Errors during strong trends (high ADX) are more costly** — the model must capture the trend direction
3. **Wrong-direction predictions should be explicitly penalised** for multi-step horizons

### Sentiment Weighting

```python
sent_weights = torch.full_like(sentiment, self.neu_w)          # Default: 0.5 (neutral)
sent_weights[sentiment < -0.05] = self.neg_w                   # Negative news: 6.0
sent_weights[sentiment >  0.05] = self.pos_w                   # Positive news: 2.0
```

The asymmetry (6× for negative vs 2× for positive vs 0.5× for neutral) reflects the well-documented **negativity bias** in financial markets: bad news causes faster and larger price reactions than equivalent good news. Neutral news carries the lowest weight since it provides little forecasting signal.

### Volatility Weighting (ADX Threshold)

```python
vol_weights[adx > self.adx_thresh] = self.vol_high_w   # Strong trend: 2.0
```

The Average Directional Index (ADX) measures trend *strength* — not direction. ADX > 25 in the original scale (converted to `thresh_25` in the scaled domain inside `train_torch_model`) indicates a well-defined directional trend. In such periods, forecasting the trend direction is essential; errors here carry 2× the weight of errors during ranging markets.

### Combined Weight Normalisation

$$w_{\text{final}} = \frac{w_{\text{sent}} \cdot w_{\text{vol}}}{\text{mean}(w_{\text{sent}} \cdot w_{\text{vol}}) + \varepsilon}$$

Dividing by the mean normalises so that the average weight is 1.0. Without this, high-weight batches (many negative-sentiment, high-ADX days) would produce systematically larger loss values, confusing the learning rate scheduler and gradient clipping thresholds.

### Weighted MSE

$$\mathcal{L}_{\text{MSE}} = \frac{1}{N}\sum_{i=1}^{N} w_i \cdot (\hat{y}_i - y_i)^2$$

### Directional Penalty (Multi-Step Only)

For horizons > 1 step, the model must also get the direction of price movement right at each step:

$$\Delta\hat{y}_t = \hat{y}_t - \hat{y}_{t-1}, \quad \Delta y_t = y_t - y_{t-1}$$

$$\mathcal{L}_{\text{dir}} = \text{ReLU}\!\left(-\Delta\hat{y}_t \cdot \Delta y_t\right) / (\sigma_{\Delta y} + \varepsilon)$$

$\text{ReLU}(-\Delta\hat{y} \cdot \Delta y)$ evaluates to 0 when both differences have the same sign (model got the direction right) and to $|\Delta\hat{y} \cdot \Delta y|$ when they have opposite signs (wrong direction). Dividing by $\sigma_{\Delta y}$ (the standard deviation of target price movements in the batch) normalises the penalty relative to typical price volatility, so it is not disproportionately large during high-volatility periods.

When `last_known` is provided (the last encoder Close price), the first difference `pred[:, 0] - last_known` uses the actual last known price rather than a synthetic zero. This makes the first-step directional penalty accurate.

**Horizon = 1 bypass:** When `pred.size(1) == 1`, no consecutive steps exist and the directional penalty cannot be computed. The code sets `combined_loss = mse_loss_raw` directly.

### Total Loss

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{MSE}} + \lambda_{\text{dir}} \cdot \mathcal{L}_{\text{dir}}$$

where $\lambda_{\text{dir}} = $ `direction_w` is tuned by Optuna in the range [0.5, 2.0].

When `reduction='none'` (used in DRO mode), the function returns the per-sample loss tensor instead of the scalar mean, allowing the training loop to group losses by regime before taking the worst-case regime mean.

---

## 18. Model Architectures

### 18.1 `TransformerEncoderLayerLayerNorm`

A custom Transformer encoder layer used exclusively by PatchTST. It differs from `EncoderLayer` in two ways:

1. **Standard MHA only** — no ProbSparse option; PatchTST uses PyTorch's `nn.MultiheadAttention`
2. **Pre-LN ordering** — LayerNorm is applied *before* the residual addition, not after:

```python
src2, _ = self.self_attn(src, src, src)
src = src + self.dropout1(src2)
src = self.norm1(src)                    # LN after residual (Post-LN)
```

Wait — looking at the code more carefully, this is actually **Post-LN** (norm after residual). But the feedforward is also Post-LN. This is the classic Transformer ordering. The class name "LayerNorm" in its name simply distinguishes it from other potential layer implementations.

This class exists separately from `EncoderLayer` because PatchTST processes `[B*C, num_patches, d_model]` tensors — the batch dimension is `B × C` because each channel is processed independently. The Informer's `EncoderLayer` processes `[B, L, d_model]` with all channels together.

### 18.2 `PatchTST`

**Source:** Nie et al., ICLR 2023 — *"A Time Series is Worth 64 Words: Long-term Forecasting with Transformers"*

#### Core Innovation: Channel-Independent Patch Processing

PatchTST processes each feature channel independently through the Transformer, treating the problem as 1D sequence modelling. This avoids learning spurious cross-channel correlations that may not generalise.

**Patch extraction:**
```python
patches = x_enc_p.unfold(dimension=2, size=patch_len, step=stride)
# [B, C, L] → [B, C, num_patches, patch_len]
```

`unfold` slides a window of size `patch_len` with step `stride` across the time dimension, extracting overlapping patches. For `seq_len=96, patch_len=16, stride=8`:

$$\text{num\_patches} = \frac{96 - 16}{8} + 1 = 11$$

**Forward pass:**
1. RevIN normalise `[B, L, C]`
2. Permute to `[B, C, L]` and extract patches → `[B, C, 11, 16]`
3. If pretraining: randomly zero 40% of patches per channel
4. Reshape to `[B·C, 11, 16]` (channel-independent batch)
5. Linear projection: `[B·C, 11, 16]` → `[B·C, 11, d_model]`
6. Add learnable positional embedding `[1, 11, d_model]`
7. Transformer encoder: $e$ layers of `TransformerEncoderLayerLayerNorm`
8. Extract channel 0 (Close price): `enc_out[:, 0, :, :]` → `[B, 11, d_model]`
9. Flatten and project: `[B, 11·d_model]` → `[B, pred_len]`
10. RevIN denormalise with `target_idx=0`

**Interpolation edge case:**
```python
if curr_patch_num != self.patch_num:
    enc_out = F.interpolate(enc_out, size=self.patch_num, mode='linear')
```
If the actual patch count differs from the expected one (can happen at dataset boundaries), the encoder output is linearly interpolated to the expected size before the projection head. This avoids dimension mismatches.

**Learnable position embedding** (vs sinusoidal in the Informer): PatchTST uses `nn.Parameter(torch.randn(1, patch_num+2, d_model) * 0.02)` — a learned position encoding that can adapt to the specific temporal structure of the training data, at the cost of not generalising beyond `patch_num` positions.

### 18.3 `Informer`

The central model. A seq2seq Transformer with ProbSparse attention, distilling encoder, and switchable normalisation.

#### Normalisation Routing

```python
if   norm_type == 'san': self.revin = SANorm(input_dim, seq_len=seq_len)
elif norm_type == 'fan': self.revin = FANorm(input_dim)
else:                    self.revin = RevIN(input_dim, affine=True, enable_revin=enable_revin)
```

The normalisation module is selected at construction time based on `norm_type`. All three modules share the same `forward(x, mode, ...)` interface, so the rest of the model code is completely unchanged regardless of which normalisation is used. This clean abstraction is what makes the ablation study possible.

#### Shared Normalisation Statistics

```python
x_enc, mean, std, gamma, beta = self.revin(x_enc, 'norm')
x_dec, _, _, _, _ = self.revin(x_dec, 'norm', mean=mean, stdev=std, gamma=gamma, beta=beta)
```

The encoder statistics are reused when normalising the decoder input. This is important: if encoder and decoder were normalised independently, they would have different zero-points, making cross-attention between them inconsistent. Sharing statistics keeps both in the same normalised space.

#### Generative Decoder (`gen_dec=True`)

```python
x    = self.dec_embedding(x_dec, x_mark_dec)
mask = torch.triu(torch.ones(L_dec, L_dec), diagonal=1).bool()
for layer in self.decoder:
    x = layer(x, enc_out, tgt_mask=mask, cross_attn=not self.dec_only)
return self.projection(self.dec_norm(x)[:, -pred_len:, :])
```

The full decoder sequence (label + future placeholder) is processed in a **single forward pass**. The causal mask enforces auto-regressive dependencies. All `pred_len` outputs are generated simultaneously from the last `pred_len` decoder positions. This is fast but trains the model to jointly predict all future steps at once.

#### Dynamic (Auto-Regressive) Decoder (`gen_dec=False`)

```python
for i in range(self.pred_len):
    curr_len = label_len + i
    x = self.dec_embedding(buf[:, :curr_len, :], ...)
    ...
    step_pred = self.projection(self.dec_norm(x)[:, -1:, :])
    preds[:, i:i+1, :] = step_pred
    buf[:, curr_len:curr_len+1, 0:1] = step_pred  # Feed back prediction
```

At inference, each step's prediction is fed back as context for the next. During training, **teacher forcing** replaces this: the ground truth target values are used as decoder context, preventing training instability from error accumulation across many steps. The `wo_Gen_Dec` ablation tests whether this auto-regressive mode outperforms or underperforms the generative mode.

### 18.4 `LSTMModel`

A bidirectional two-layer LSTM with RevIN normalisation. The LSTM gate equations:

$$f_t = \sigma(W_f [h_{t-1}, x_t] + b_f) \quad \text{(forget gate — what to erase from cell)}$$
$$i_t = \sigma(W_i [h_{t-1}, x_t] + b_i) \quad \text{(input gate — what new info to write)}$$
$$\tilde{c}_t = \tanh(W_c [h_{t-1}, x_t] + b_c) \quad \text{(candidate cell values)}$$
$$c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t \quad \text{(cell state update)}$$
$$o_t = \sigma(W_o [h_{t-1}, x_t] + b_o) \quad \text{(output gate — what to expose)}$$
$$h_t = o_t \odot \tanh(c_t) \quad \text{(hidden state)}$$

The cell state $c_t$ is the LSTM's "long-term memory" — the forget gate selectively erases old information while the input gate selectively writes new information. This explicit memory mechanism is how LSTMs handle sequences where distant past events (e.g., a bear market 60 days ago) still influence the current price.

**Forward pass:** RevIN normalise → LSTM processes full sequence → take only the **last hidden state** `out[:, -1, :]` as a fixed-size sequence summary → linear projection to `pred_len` outputs → RevIN denormalise.

Multi-layer dropout is only applied between layers — `dropout=0` is forced when `num_layers=1` because PyTorch's LSTM implementation does not support dropout with a single layer.

### 18.5 `TCNModel`

Stacks `TemporalBlock` layers with exponentially increasing dilation:

```python
for i in range(len(num_channels)):
    dilation_size = 2 ** i          # 1, 2, 4, 8, ...
    padding = (kernel_size - 1) * dilation_size
    layers += [TemporalBlock(in_ch, num_channels[i], kernel_size, 1,
                              dilation_size, padding, dropout)]
```

**Forward pass:** RevIN normalise → permute `[B, L, C]` → `[B, C, L]` → TCN stack → extract last timestep `out[:, :, -1]` → linear projection to `pred_len` → RevIN denormalise.

The extraction of `out[:, :, -1]` (the last timestep's output) works because the dilated causal convolutions have already integrated all lookback information into the final position — the last output summarises the entire sequence within the receptive field.

### 18.6 `EnsembleModel`

A weighted linear combination of four trained models plus a naive persistence baseline:

$$\hat{y}_{\text{ensemble}} = w_0\hat{y}_{\text{Informer}} + w_1\hat{y}_{\text{PatchTST}} + w_2\hat{y}_{\text{LSTM}} + w_3\hat{y}_{\text{TCN}} + w_4\hat{y}_{\text{Naive}}$$

**Naive forecast:**
```python
p_naive = enc[:, -1:, 0:1].repeat(1, self.pred_len, 1)
```
The last known Close price in the encoder input is repeated for all `pred_len` future steps. Including this in the ensemble hedges against all deep learning models failing simultaneously during structural breaks or market crashes.

**Parameter freezing:**
```python
for model in self.models.values():
    for param in model.parameters():
        param.requires_grad = False
```
All submodel parameters are frozen. The ensemble is not end-to-end trainable — weights are computed analytically from validation performance. This preserves the individually-optimised characteristics of each submodel.

**Two ensemble configurations:**
- **Equal Weight** — `[0.20, 0.20, 0.20, 0.20, 0.20]` — treats all models identically; a simple sanity check
- **Dynamic Inverse-MSE** — `w_i = (1/\text{MSE}_i) / \sum_j (1/\text{MSE}_j)` — gives higher weight to models that performed better on the validation set while still maintaining diversification

---

## 19. Training Infrastructure

### 19.1 `EarlyStopping`

```python
def __call__(self, val_loss):
    score = -val_loss            # Negate: higher score = lower loss = better
    if self.best_score is None:
        self.best_score = score
    elif score < self.best_score:
        self.counter += 1        # No improvement
        if self.counter >= self.patience:
            self.early_stop = True
    else:
        self.best_score = score  # New best — reset counter
        self.counter = 0
```

The counter is incremented every epoch that does not achieve a new best validation MSE, and reset whenever a new best is found. `patience=20` gives the model 20 consecutive chances to recover before halting — this tolerates temporary plateaus (common during learning rate reduction) while preventing indefinite overfitting. The `best_score` is negated because higher scores correspond to lower losses, making the comparison `score < self.best_score` equivalent to `loss > best_loss`.

### 19.2 `train_torch_model`

The main training loop supporting both standard ERM and DRO training.

#### ADX Threshold Rescaling

```python
cov_idx = adx_idx - 1   # Covariate index (excluding target column)
adx_min = scaler_cov.data_min_[cov_idx]
adx_max = scaler_cov.data_max_[cov_idx]
thresh_25 = (25.0 - adx_min) / (adx_max - adx_min + 1e-8)
```

The ADX threshold of 25 (standard technical analysis convention for "strong trend") is converted from the original ADX scale to the MinMax-scaled domain. `MultiFactorLoss` operates on scaled values, so without this conversion the threshold comparison `adx > self.adx_thresh` would be comparing against the wrong scale.

#### Optimiser — Adam

$$m_t = \beta_1 m_{t-1} + (1-\beta_1)g_t \quad \text{(first moment — gradient direction)}$$
$$v_t = \beta_2 v_{t-1} + (1-\beta_2)g_t^2 \quad \text{(second moment — gradient magnitude)}$$
$$\theta_t = \theta_{t-1} - \frac{\eta}{\sqrt{\hat{v}_t} + \varepsilon}\hat{m}_t \quad \text{(parameter update)}$$

Adam adapts the learning rate individually for each parameter based on the historical gradient magnitudes. Parameters with consistently large gradients get smaller effective learning rates; parameters with small gradients get larger effective learning rates. This is especially useful in models with very different gradient scales (e.g., the embedding layers vs the projection head).

`weight_decay` adds an L2 penalty: $\mathcal{L}_{\text{reg}} = \mathcal{L} + \lambda\|\theta\|^2$. This penalises large weights, acting as a regulariser that prevents overfitting.

#### Scheduler — ReduceLROnPlateau

```python
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=3
)
```

Halves the learning rate after 3 consecutive validation epochs without improvement. This allows the model to escape local regions where the gradient is large and noisy (high LR helpful) and then fine-tune near the optimum (low LR needed). The `factor=0.5` reduction is conservative — after 3 reductions the LR is $1/8$ of its original value.

#### DRO Training Mode

```python
regime_losses = []
for i in range(3):
    mask = (batch_regimes == i)
    if mask.any():
        regime_losses.append(raw_losses[mask].mean())
loss = torch.stack(regime_losses).max()
```

**Distributionally Robust Optimisation (DRO)** minimises the worst-case loss across the three GMM market regimes instead of the average:

$$\min_\theta \max_{k \in \{0,1,2\}} \mathbb{E}[\mathcal{L}(\theta, x, y) \mid \text{regime} = k]$$

Standard ERM (averaging losses) allows the model to sacrifice performance in one regime if it gains enough in the others. DRO prevents this by forcing the model to perform at least reasonably in its weakest regime. This is particularly important for stock forecasting because the test period may have a different regime distribution than the training period.

`reduction='none'` mode returns per-sample losses, which are then partitioned by regime label before taking the per-regime mean and overall maximum.

#### NaN/Inf Guard

```python
loss.backward()
if torch.isnan(loss) or torch.isinf(loss):
    return float('inf')
```

The check is performed **after** `loss.backward()` but **before** `optimizer.step()`. This ensures corrupted gradients do not update the model weights. Returning `float('inf')` signals to Optuna that this trial should be discarded (worse than any valid trial).

#### Gradient Clipping

```python
grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

If the global L2 norm of all parameter gradients exceeds 1.0, all gradients are scaled down uniformly so the norm equals exactly 1.0. The pre-clipping norm is returned and logged — monitoring this value can reveal training instabilities early.

#### Validation and Logging

```python
if (epoch + 1) % 5 == 0 or epoch == 0:
    print(f"Epoch [{epoch+1}/{epochs}] | Val Scaled MSE: {val_mse_scaled:.6f}%")
```

Printing every 5 epochs balances visibility with log verbosity for runs of up to 300 epochs. When not inside an Optuna trial (i.e., the final training run after hyperparameter selection), all four metrics are logged to MLflow per epoch, enabling learning curve visualisation in DagsHub.

```python
if trial is not None:
    trial.report(val_mse_scaled, epoch)
    if trial.should_prune():
        raise optuna.TrialPruned()
```

During Optuna trials, the validation MSE is reported at each epoch. Optuna's MedianPruner compares this to the median performance of previous trials at the same epoch — if the current trial is performing significantly worse, `should_prune()` returns `True` and `TrialPruned` is raised, terminating the trial early and freeing resources for more promising hyperparameter combinations.

---

## 20. Metric & Evaluation Helpers

### `calculate_metrics`

```python
real_pred = scaler_target.inverse_transform(flat_preds).reshape(orig_shape)
real_act  = scaler_target.inverse_transform(flat_actuals).reshape(orig_shape)

mse_real   = mean_squared_error(real_act.flatten(), real_pred.flatten())
mae_real   = mean_absolute_error(real_act.flatten(), real_pred.flatten())
mse_scaled = mean_squared_error(actuals_scaled.flatten(), preds_scaled.flatten()) * 100
mae_scaled = mean_absolute_error(actuals_scaled.flatten(), preds_scaled.flatten()) * 100
```

Four metrics are computed:

| Metric | Formula | Unit | Purpose |
|--------|---------|------|---------|
| `mse_real` | $\frac{1}{N}\sum(\hat{y}-y)^2$ | USD² | Absolute magnitude of errors |
| `mae_real` | $\frac{1}{N}\sum|\hat{y}-y|$ | USD | Human-interpretable error |
| `mse_scaled` | $\frac{100}{N}\sum(\hat{y}_s - y_s)^2$ | % | Comparable across price levels |
| `mae_scaled` | $\frac{100}{N}\sum|\hat{y}_s - y_s|$ | % | Comparable across price levels |

The `× 100` for scaled metrics converts raw MinMax-domain values (typically $10^{-5}$ to $10^{-3}$) to percentages of the full scaled range, making them much more human-readable. The **NaN/Inf guard** returns `float('inf')` across all metrics immediately if any prediction is invalid, causing Optuna to prune the trial and preventing corrupt metrics from being stored.

### `calculate_naive_metrics`

Computes persistence forecast metrics starting from `fixed_start_idx` (the overlap boundary). This pure-Python version exists separately from `evaluate_naive_baseline` because it is called internally by `evaluate_ensembles` to compute the naive model's validation MSE for the inverse-MSE weight calculation, without the side effects of logging to MLflow or printing formatted output.

---

## 21. Pipeline Orchestration Functions

### `setup_scenario_context`

Routes data and configuration based on the active scenario:

```python
if scenario == "Without Sentiment":
    train_data = train_vals_full[:, :-1]   # Drop the last column (Sentiment_Score)
    use_weighted_loss = False              # Use plain MSELoss
    current_sent_idx  = None
    # Adjust ADX index since one column was removed
    current_adx_idx = adx_idx_full - 1 if adx_idx_full > sent_idx_full else adx_idx_full
```

Sentiment is always appended as the last column during data loading, so `[:, :-1]` cleanly removes it. When sentiment is removed, `MultiFactorLoss` cannot be used (it requires sentiment to assign weights), so `use_weighted_loss = False` forces plain MSE. The ADX column index must be adjusted because removing a column shifts subsequent column indices by -1.

### `evaluate_naive_baseline`

Computes the persistence forecast — the simplest possible model:

$$\hat{y}_{t+1} = \hat{y}_{t+2} = \ldots = \hat{y}_{t+H} = y_t$$

Today's price repeated for all future steps. Every model must beat this baseline; any model that cannot outperform simple persistence is providing less information than doing nothing. Results are logged to MLflow (with only `Runner` and `Track` tags) and saved to the per-scenario/horizon Excel file. The function now takes `OUTPUT_FILE` as a parameter (instead of reading a global) because the output path is dynamically computed in `main()`.

### `perform_hyperparameter_tuning`

Runs 100 Optuna trials for a given model. Each trial is a **nested MLflow run** (child of the current main run), tagged with trial number, model name, scenario, and horizon.

**Parameter sharing via caching** in `process_standalone_models`:
```python
if is_informer_type and inf_params_cache is not None:
    best_params, opt_epoch = inf_params_cache, inf_epoch_cache
```
All 19 Informer ablation variants share the hyperparameters tuned for `Informer_Base`. This saves 18 × 100 = 1,800 tuning trials (roughly 180 hours of compute at 6 min/trial). The justification: the main differences between ablation variants (norm type, distil, prob_sparse) are architectural, not dependent on the learning rate or dropout level. Similarly, `PatchTST_Pretrained` reuses `PatchTST_Standalone`'s params.

**Norm type detection inside Optuna:**
```python
n_type = 'revin'
if 'SANorm' in model_name: n_type = 'san'
elif 'FANorm' in model_name: n_type = 'fan'
```
The model name string is inspected during hyperparameter tuning to select the correct normalisation, since the first Informer variant to be tuned sets the cached params for all others.

### `prepare_window_data`

```python
full_train_vals = np.concatenate([train_data, val_data[val_overlap_len:]], axis=0)
```

For the final training run (after hyperparameter tuning), the validation data (minus its overlap with training) is appended to the training set. This maximises data utilisation — the model trains on everything up to the test boundary. The test data is trimmed to start exactly `lookback` steps before the first valid prediction position:

```python
trim_start = test_overlap_len - lookback
current_test_data = test_data[trim_start:]
```

### `process_standalone_models` — Phase 1

Iterates over all 23 models in `standalone_configs`. For each model:

1. **Checkpoint check** → skip if MLflow shows a finished run for this runner
2. **Dataset creation** via `get_model_datasets` (routes to `InformerDataset` or `PatchTSTDataset`)
3. **Hyperparameter tuning** (or cache hit)
4. **Architecture construction** using `best_params`
5. **TTA weight loading** — if `tta=True`, loads base model weights instead of training from scratch
6. **Training** via `train_torch_model` with appropriate flags (`use_dro`, loss weights)
7. **MC-Dropout evaluation** — 10 forward passes with model in `train()` mode (dropout active)
8. **TTA adaptation** — for TTA models, adapts RevIN parameters on each test batch before prediction
9. **PICP computation** — Prediction Interval Coverage Probability of the 95% MC-Dropout interval
10. **Logging** to MLflow + Excel

**MC-Dropout rationale:** Running the model with dropout active during inference makes each forward pass stochastically different (different neurons are dropped each time). The mean across 10 passes is the final prediction; the standard deviation is an uncertainty estimate. PICP measures what fraction of true values fall within the $[2.5\%, 97.5\%]$ percentile bounds of the 10 samples — a well-calibrated model should achieve PICP ≈ 0.95.

**TTA mechanics:**
```python
for name, param in tta_model.named_parameters():
    param.requires_grad = 'revin' in name   # Only adapt RevIN affine params
```
Only `affine_weight` and `affine_bias` in the RevIN module are updated during TTA. All other weights are frozen. For each test batch, a self-supervised task is applied: the encoder input is time-shifted by `pred_len` positions (with zeros filling the gap), and the model is asked to predict the original last `pred_len` steps of the encoder. This forces the RevIN module to adapt its scale and shift to the test batch's local statistics before making the actual prediction — like instance-level fine-tuning at test time.

### `evaluate_ensembles` — Phase 2

```python
inv_mses        = [1.0 / (val_mses[m] + 1e-9) for m in all_names]
total_inv       = sum(inv_mses)
dynamic_weights = [inv / total_inv for inv in inv_mses]
```

Computes validation MSEs for the best Informer variant, best PatchTST variant, LSTM, TCN, and naive baseline. The inverse-MSE weighting gives higher weight to models that performed better on the validation set. The $10^{-9}$ prevents division by zero if a model achieves perfect validation performance. Two ensemble variants (Equal Weight and Dynamic InvMSE) are then evaluated on the test set.

### `save_overall_best_model`

Saves both the `state_dict` and the full model object for the best-performing model in the window:

```python
mlflow.pytorch.log_model(pytorch_model=best_overall_model,
                         artifact_path="model",
                         registered_model_name=f"Champion_{clean_scenario}_H{pred_len}")
```

The MLflow Model Registry maintains versioned model artifacts. Each new run that calls `log_model` with the same `registered_model_name` creates a new version rather than overwriting the previous one, maintaining a full history. A summary Excel file records the champion model alongside naive baseline metrics for easy comparison.

### `main`

The entry point loops over all scenario × horizon combinations:

```python
for scenario in scenarios:          # ["With Sentiment", "Without Sentiment"]
    for pred_len in prediction_horizons:   # [1, 8, 16]
        for lookback in lookback_windows:  # [96, 32, 16]
```

Data is loaded **once** before all loops (`load_and_split_data`) and the scenario-specific view is created at the scenario level (`setup_scenario_context`), avoiding redundant file I/O. After all configurations complete, a final leaderboard is printed showing every model's Scaled MSE across all 18 configurations.

---

## 22. Ablation Suite — Model Name Reference

### Informer Variants (19 models)

| Model Name | Norm | Distil | ProbSparse | DecOnly | GenDec | Special |
|------------|------|--------|-----------|---------|--------|---------|
| `Informer_Base` | RevIN | ✅ | ✅ | ❌ | ✅ | Baseline |
| `Informer_Base_TTA` | RevIN | ✅ | ✅ | ❌ | ✅ | Test-Time Adapt |
| `Informer_DRO` | RevIN | ✅ | ✅ | ❌ | ✅ | DRO training |
| `Informer_SANorm` | SANorm | ✅ | ✅ | ❌ | ✅ | — |
| `Informer_FANorm` | FANorm | ✅ | ✅ | ❌ | ✅ | — |
| `Informer_FANorm_TTA` | FANorm | ✅ | ✅ | ❌ | ✅ | TTA |
| `Informer_NoTemporal` | RevIN | ✅ | ✅ | ❌ | ✅ | No calendar embed |
| `Informer_NoPositional` | RevIN | ✅ | ✅ | ❌ | ✅ | No positional enc |
| `Informer_NoBothEmbeds` | RevIN | ✅ | ✅ | ❌ | ✅ | Value embed only |
| `Informer_DecOnly` | RevIN | ✅ | ✅ | ✅ | ✅ | No encoder |
| `Informer_FullAttn` | RevIN | ✅ | ❌ | ❌ | ✅ | Standard MHA |
| `Informer_NoDistil` | RevIN | ❌ | ✅ | ❌ | ✅ | No downsampling |
| `Informer_NoProb_NoDistil` | RevIN | ❌ | ❌ | ❌ | ✅ | Vanilla Transformer |
| `wo_Gen_Dec` | RevIN | ✅ | ✅ | ❌ | ❌ | Auto-regressive dec |
| `Informer_SANorm_NoTemporal` | SANorm | ✅ | ✅ | ❌ | ✅ | No calendar |
| `Informer_SANorm_DecOnly` | SANorm | ✅ | ✅ | ✅ | ✅ | No encoder |
| `Informer_FANorm_NoTemporal` | FANorm | ✅ | ✅ | ❌ | ✅ | No calendar |
| `Informer_FANorm_DecOnly` | FANorm | ✅ | ✅ | ✅ | ✅ | No encoder |
| `Informer_FANorm_DecOnly_NoTemporal` | FANorm | ✅ | ✅ | ✅ | ✅ | No enc, no cal |

### Other Standalone Models (4 models)

| Model Name | Architecture | Notes |
|------------|-------------|-------|
| `PatchTST_Standalone` | Patch Transformer | Supervised only |
| `PatchTST_Pretrained` | Patch Transformer | MPR pretraining + fine-tune |
| `LSTM_Standalone` | 2-layer LSTM | With RevIN |
| `TCN_Standalone` | Dilated causal TCN | With RevIN |

### Ensemble Models (2 models)

| Model Name | Description |
|------------|-------------|
| `Ensemble_Equal_Weight` | All 5 components at weight 0.20 |
| `Ensemble_Dynamic_InvMSE` | Weights $\propto 1/\text{val\_MSE}$ |

**Total: 25 distinct models evaluated per scenario × horizon × lookback combination.**

---

## 23. Data Flow Diagram

```
CSV: aapl_stock_sentiment_new_dataset_2017_2026.csv
   │
   ▼
load_and_split_data()
   ├── Sort by Date
   ├── Average sentiment per date (multi-row CSV handling)
   ├── Sentiment shift(1) — prevent look-ahead leakage
   ├── 60/10/30 chronological split
   ├── Extend val and test slices by max_lookback (96) for warm-up context
   ├── GMM fit on training volatility only (3 regimes)
   ├── Forward-fill → backward-fill (train only) → zero-fill
   ├── scaler_cov.fit_transform(train covariates only)
   ├── scaler_target.fit_transform(train Close only)
   ├── Restore raw sentiment scores (bypass MinMax)
   └── Returns: train/val/test arrays, marks, regimes, scalers, col indices

   ↓
for scenario in ["With Sentiment", "Without Sentiment"]:

   setup_scenario_context()
   ├── "With Sentiment"    → keep all cols, use MultiFactorLoss
   └── "Without Sentiment" → drop sentiment col, use MSELoss

   for pred_len in [1, 8, 16]:

       evaluate_naive_baseline()     ← Persistence forecast lower bound

       for lookback in [96, 32, 16]:

           prepare_window_data()
           ├── Extend training with val non-overlap rows
           └── Trim test to start exactly lookback before boundary

           ┌─── PHASE 1: process_standalone_models() ──────────────────┐
           │                                                             │
           │  for model_name in standalone_configs (23 models):         │
           │    1. is_completed()? → query MLflow DagsHub               │
           │       YES → skip. NO → continue.                           │
           │                                                             │
           │    2. get_model_datasets()                                  │
           │       → InformerDataset (9-tuple) or PatchTSTDataset (5)   │
           │                                                             │
           │    3. Hyperparameter tuning (or cache hit)                  │
           │       perform_hyperparameter_tuning()                       │
           │         └── 100 Optuna Bayesian trials                     │
           │              ├── train_torch_model() (with val set)        │
           │              ├── trial.report() → Optuna pruning           │
           │              └── best_params, best_epoch stored            │
           │                                                             │
           │    4. Architecture construction from best_params            │
           │                                                             │
           │    5. [pretrain_patchtst_model()] if is_pretrain           │
           │       └── Masked Patch Reconstruction, 40% mask ratio      │
           │                                                             │
           │    6. train_torch_model() — FINAL TRAINING                 │
           │       ├── Adam + ReduceLROnPlateau                         │
           │       ├── MultiFactorLoss or MSELoss (ENABLE_CUSTOM_LOSS)  │
           │       ├── DRO: max(regime_losses) if dro=True              │
           │       ├── Gradient clipping (max_norm=1.0)                 │
           │       ├── Early stopping (patience=20)                     │
           │       └── MLflow per-epoch logging                         │
           │                                                             │
           │    7. MC-Dropout: 10 forward passes, model.train()         │
           │       └── mean = prediction, std = uncertainty             │
           │                                                             │
           │    8. [TTA]: adapt RevIN params on each test batch          │
           │       └── Self-supervised: predict shifted encoder input   │
           │                                                             │
           │    9. PICP = P(actual ∈ [2.5%, 97.5%] of MC samples)      │
           │                                                             │
           │   10. Log to MLflow + save to Excel                        │
           │                                                             │
           └─────────────────────────────────────────────────────────────┘

           ┌─── PHASE 2: evaluate_ensembles() ──────────────────────────┐
           │                                                             │
           │  1. Re-evaluate all 4 best DL models on validation set      │
           │  2. Compute naive val MSE                                   │
           │  3. Dynamic weights: w_i = (1/MSE_i) / Σ(1/MSE_j)        │
           │  4. Evaluate Ensemble_Equal_Weight on test set              │
           │  5. Evaluate Ensemble_Dynamic_InvMSE on test set           │
           │  6. Log to MLflow + save to Excel                          │
           │                                                             │
           └─────────────────────────────────────────────────────────────┘

           save_overall_best_model()
           ├── torch.save(state_dict) + torch.save(full model)
           ├── mlflow.pytorch.log_model → MLflow Model Registry
           └── Best_Models_Summary Excel file

           torch.cuda.empty_cache() + gc.collect()

Final leaderboard printed across all 18 configurations
```

---

*Documentation generated for the pipeline codebase — multi-scenario, multi-horizon, MLflow DagsHub checkpoint system, ANSI-native dual-mode logging. Normalisation methods: RevIN (Kim et al. ICLR 2022), SANorm (Liu et al. NeurIPS 2023), FANorm (arXiv 2409.20371).*
