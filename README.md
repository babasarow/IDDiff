# IDDiff: Intent-aware Deterministic Diffusion Model for Next Location Prediction

This repository provides the PyTorch implementation of **IDDiff**, an intent-aware deterministic diffusion framework for next location prediction.

IDDiff contains three main components:

1. **Sequence-aware Trajectory Representation Module** constructs a whole-trajectory representation, extracts local intent prototypes through sliding-window segmentation and DBSCAN clustering, and selects the dominant prototype as the diffusion condition.
2. **Location Distance Graph Encoder** captures geographical dependencies among Locations with distance-aware graph convolution.
3. **Intent Refinement Module** generates a location archetype and refines it through an intent-guided deterministic diffusion process.

## 📦 Environment

### 1. Clone the repository

```bash
git clone <repository-url>
cd IDDiff
```

### 2. Create a Conda environment

```bash
conda create -n iddiff python=3.10 -y
conda activate iddiff
```

### 3. Install dependencies

Install a PyTorch build compatible with your CUDA version, and then install the remaining packages:

```bash
pip install torch torchvision torchaudio
pip install torch-geometric numpy pandas scikit-learn
```

> **Note:** For GPU execution, the installed PyTorch and PyTorch Geometric versions must be compatible with the local CUDA toolkit.

## 🗂️ Project Structure

```text
IDDiff/
├── main.py               # Training and full-ranking evaluation entry point
├── model.py              # Main IDDiff architecture
├── layers.py             # Distance-aware GCN and deterministic diffusion layers
├── dataset.py            # Dataset loading, graph construction, and metrics
├── parse.py              # Command-line arguments
├── gol.py                # Runtime configuration, paths, and global settings
├── cluster_utils.py      # Sliding-window and DBSCAN utilities
├── sequence_encoder.py   # Attention-PFFN sequence encoder
├── model_refactor.py     # Modular trajectory-representation implementation
├── config.py             # Configuration for the refactored modules
```

## 🔧 Data Preparation

### Step 1: Configure paths

Before running the code, edit the following variables in `gol.py`:

```python
DATA_PATH = "/path/to/data/processed"
FILE_PATH = "/path/to/checkpoints"
```

`DATA_PATH` should contain one subdirectory for each dataset:

```text
data/processed/
├── IST/
├── JK/
├── SP/
├── NYC/
└── LA/
```

### Step 2: Prepare the processed files

Each dataset directory must contain the following files:

```text
<DATASET>/
├── all_data.pkl
├── dist_mat.npy
├── dist_graph.pkl
└── dist_on_graph.npy
```

Their roles are:

- `all_data.pkl`: user/location statistics, trajectory instances, and chronological training, validation, and test splits.
- `dist_mat.npy`: pairwise location distance matrix used for spatio-temporal interval construction.
- `dist_graph.pkl`: edge indices of the global location distance graph.
- `dist_on_graph.npy`: distance-based edge values associated with the location graph.

> **Note:** Data preprocessing scripts and raw datasets are not included in the current code package. The processed files must follow the structure expected by `dataset.py`.

## 🚀 Training and Evaluation

### Step 1: Train IDDiff

The reported hyperparameter setting uses an embedding dimension of `128`, the Adam optimizer with a learning rate of `5e-4`, a balance coefficient of `0.75`, a diffusion-loss coefficient of `0.2`, a dropout rate of `0.25`, a sliding-window size of `4`, and `100` diffusion steps. In the current command-line interface, the balance and diffusion-loss coefficients correspond to `--alpha` and `--zeta`, respectively.

The following command provides an example configuration for training on the LA dataset:

```bash
python main.py \
  --dataset LA \
  --batch 128 \
  --epoch 100 \
  --lr 0.0005 \
  --decay 0.001 \
  --hidden 128 \
  --num_heads 4 \
  --layer 2 \
  --length 100 \
  --window_size 4 \
  --dbscan_eps 0.60 \
  --dbscan_min_samples 3 \
  --guidance_w 0.30 \
  --dropout \
  --dp 0.25 \
  --diffsize 100 \
  --interp_steps 100 \
  --sample_num 15 \
  --alpha 0.75 \
  --zeta 0.20 \
  --patience 10 \
  --gpu 0 \
  --save
```

Available dataset names are:

```text
IST, JK, SP, NYC, LA
```

The model performs validation after each epoch and selects the best checkpoint according to **Recall@5**. When `--save` is enabled, the checkpoint is stored as:

```text
<FILE_PATH>/weight.pth
```

### Step 2: Evaluate a saved checkpoint

Use `--load` to load `weight.pth` before evaluation and training:

```bash
python main.py --dataset LA --gpu 0 --load
```

The evaluation procedure reports ranking performance at multiple cutoffs, including:

- Acc@1
- Recall@1/2/5/10/20
- NDCG@1/2/5/10/20
- MRR


