# BrainFlow

MindEyeV2 multi-subject fMRI dataloader with multi-GPU Distributed Data Parallel (DDP) support, tensor caching, and async prefetching. All tunable parameters live in `config.yaml` — no code changes needed.

## Quick Start

1. Create a `.env` file with your HuggingFace token:
   ```
   HF_TOKEN=hf_your_token_here
   ```

2. Edit `config.yaml` to match your hardware (number of GPUs, batch size, etc.)

3. Launch:
   ```powershell
   # Multi-GPU — nproc_per_node must match config.yaml → distribution.num_gpus
   python launch.py --nproc_per_node=2 dataloader.py

   # Single-GPU / CPU
   python dataloader.py
   ```

## Configuration (`config.yaml`)

Every parameter is controlled from `config.yaml`. The dataloader reads it at startup — nothing is hard-coded in `dataloader.py`.

### Distribution

```yaml
distribution:
  num_gpus: 2            # Number of GPUs (or CPU workers) to use
  backend: auto          # "auto" → nccl if CUDA, gloo otherwise | "nccl" | "gloo"
  init_method: "env://"  # Reads MASTER_ADDR/PORT/RANK/WORLD_SIZE from environment
```

### Data

```yaml
data:
  hf_repo: "pscotti/mindeyev2"
  data_dir: "./mindeyev2_cache"
  tensor_cache: "all_subjects_tensors.pt"
  subjects: [1, 2, 3, 4, 5, 6, 7, 8]   # Any subset of [1..8]
  img_size: 256
  max_train: 8859
  max_test: 982
  download_threads: 4                    # Parallel threads per subject
  force_rebuild: false                   # Ignore cache and re-download
```

### DataLoader

```yaml
dataloader:
  batch_size_per_gpu: 32   # Effective batch = this × num_gpus
  num_workers: 4           # CPU workers per GPU (0 = main process)
  prefetch_factor: 2       # Batches prefetched per worker
  pin_memory: true         # Page-locked memory for async GPU transfers
  persistent_workers: true # Keep workers alive between epochs
  drop_last_train: true    # Drop incomplete final batch (training)
  drop_last_test: false    # Keep incomplete final batch (evaluation)
```

### Training

```yaml
training:
  num_epochs: 3
  log_every_n_steps: 50    # Print progress every N steps (rank 0 only)
```

## Data Source

All data is downloaded from [`pscotti/mindeyev2`](https://huggingface.co/datasets/pscotti/mindeyev2) on HuggingFace.

### Per-Subject Files

| File | Size (approx.) | Description |
|------|-----------------|-------------|
| `betas_all_subj0X_fp32_renorm.hdf5` | ~1.9 GB | fMRI beta weights (voxel activations) for all trials |
| `wds/subj0X/train/{0..7}.tar` | ~8 MB each | WebDataset shards with behavioural metadata (train split) |
| `wds/subj0X/test/0.tar` | ~35 MB | WebDataset shard with behavioural metadata (test split) |
| `COCO_73k_subj_indices.hdf5` | shared | COCO image stimuli mapped to subject trial indices |

With all 8 subjects, the total raw download is **~16 GB**.

## Loading Pipeline

### First Run (slow path)

Only happens once (or when `force_rebuild: true`). Rank 0 does all the work; other ranks wait at a barrier.

```
For each subject in config.subjects:
  1. Download files from HuggingFace (download_threads per subject)
     ├── betas HDF5       (~1.9 GB)
     ├── 8 train shards   (~70 MB total)
     └── 1 test shard     (~35 MB)
  2. Read behavioural metadata from WebDataset .tar shards
     └── Extract behav.npy from each sample
  3. Build trial/COCO index arrays from behavioural data
     ├── Train: up to max_train trials per subject
     └── Test:  up to max_test trials per subject
  4. Load fMRI betas via vectorized unique-index reads
     └── HDF5 with 256 MB chunk cache for fast I/O
  5. Load + preprocess COCO images
     ├── Unique-index read from HDF5
     ├── Normalize to [0, 1]
     └── Resize to img_size × img_size (bilinear interpolation)
```

After all subjects are loaded, everything is saved to:

```
<data_dir>/<tensor_cache>  (~12–15 GB for 8 subjects)
```

### Subsequent Runs (fast path)

```
torch.load("<data_dir>/<tensor_cache>")
```

Each rank loads the cache independently. No downloads, no HDF5 reads. To force a fresh download, set `force_rebuild: true` in `config.yaml`.

## GPU Distribution (DDP)

### Architecture

```
                    ┌──────────────────────────────────────┐
                    │          launch.py (launcher)         │
                    │  Patches TCPStore (libuv workaround)  │
                    │  Spawns N worker processes             │
                    └──────────┬───────────────────────────┘
                               │
        ┌──────────┬───────────┼───────────┬──────────┐
        ▼          ▼           ▼           ▼          ▼
    ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐     ...
    │ Rank 0 │ │ Rank 1 │ │ Rank 2 │ │ Rank 3 │  (up to N)
    │ GPU 0  │ │ GPU 1  │ │ GPU 2  │ │ GPU 3  │
    └────────┘ └────────┘ └────────┘ └────────┘
```

Each rank runs a full copy of `dataloader.py`, reading from the same `config.yaml`. PyTorch DDP handles gradient synchronization across ranks.

### How Data is Split Across GPUs

`DistributedSampler` partitions the combined dataset (all configured subjects concatenated) so that **each GPU sees a unique, non-overlapping slice**:

```
Example: 8 subjects × 8,859 train samples = 70,872 total

With num_gpus: 2
  GPU 0 → samples [0, 2, 4, ...]  = 35,436 samples
  GPU 1 → samples [1, 3, 5, ...]  = 35,436 samples

With num_gpus: 8
  GPU 0 → samples [0, 8, 16, ...]  = 8,859 samples
  GPU 1 → samples [1, 9, 17, ...]  = 8,859 samples
  ...
  GPU 7 → samples [7, 15, 23, ...] = 8,859 samples
```

No sample is seen by more than one GPU in a given epoch.

### Batch Sizes

All values come from `config.yaml`:

| Setting | Config key | Example |
|---------|-----------|---------|
| Per-GPU batch size | `dataloader.batch_size_per_gpu` | 32 |
| Number of GPUs | `distribution.num_gpus` | 2 |
| **Effective batch size** | auto-computed | **64** |
| Train batches per GPU | auto-computed | ~1,108 |

### Process Group Setup

- **Backend**: Controlled by `distribution.backend`. When set to `auto`, uses `nccl` if CUDA is available, otherwise `gloo`.
- **Init method**: Controlled by `distribution.init_method`. Default `env://` reads `MASTER_ADDR`, `MASTER_PORT`, `RANK`, `WORLD_SIZE` from the environment (set automatically by `launch.py`).
- Each process calls `torch.cuda.set_device(local_rank)` to bind to its assigned GPU.

### Epoch Reseeding

```python
train_sampler.set_epoch(epoch)
```

Called at the start of every epoch. This changes the random seed used by `DistributedSampler` so each GPU gets a **different permutation** of its data slice each epoch, preventing the model from memorizing sample order.

## DataLoader Performance Features

All features are toggled via `config.yaml`:

| Feature | Config key | What It Does |
|---------|-----------|--------------|
| Pin memory | `dataloader.pin_memory` | Page-locked CPU memory for async DMA transfers to GPU |
| Non-blocking transfer | (in training loop) | `.to(device, non_blocking=True)` overlaps transfer with compute |
| Persistent workers | `dataloader.persistent_workers` | Workers stay alive between epochs (no respawn overhead) |
| Prefetch factor | `dataloader.prefetch_factor` | Each worker prefetches N batches ahead, keeping the GPU fed |
| Custom collate | `fast_collate()` | Stacks tensors directly, skipping default collate overhead |
| Pinned tensors in Dataset | `dataloader.pin_memory` | `fmri` and `images` tensors are pinned at construction time |

### Data Transfer Timeline (per training step)

```
CPU Worker:  [load batch N+2] [load batch N+3] ...
DMA:              [transfer batch N+1 → GPU]
GPU:                    [forward/backward on batch N]
```

Prefetching + pinned memory + non-blocking transfers keep all three pipelines running concurrently.

## Windows Notes

- **libuv**: PyTorch's `TCPStore` defaults to `use_libuv=True`, but Windows CPU builds lack libuv support. `launch.py` and `dataloader.py` both monkey-patch `TCPStore` to set `use_libuv=False`.
- **NCCL**: Not available in CPU-only builds. Set `backend: auto` (default) and it falls back to `gloo` automatically.
- **Signals**: `SIGHUP`/`SIGQUIT` warnings on launch are harmless — those signals don't exist on Windows.

## Project Structure

```
BrainFlow/
├── config.yaml                # All tunable parameters (edit this)
├── dataloader.py              # Main dataloader with DDP support
├── launch.py                  # torchrun wrapper (Windows libuv fix)
├── requirements.txt           # Python dependencies
├── .env                       # HF_TOKEN (not committed)
└── mindeyev2_cache/           # Auto-created on first run
    ├── COCO_73k_subj_indices.hdf5
    ├── betas_all_subj0X_fp32_renorm.hdf5  (×8)
    ├── wds/subj0X/train/{0..7}.tar        (×8)
    ├── wds/subj0X/test/0.tar              (×8)
    └── all_subjects_tensors.pt            # Tensor cache (~12-15 GB)
```