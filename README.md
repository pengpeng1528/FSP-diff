# FSP-diff

This repository uses a two-stage training pipeline:

- `trainS1.sh`: Stage 1 training (`FSPS1`)
- `trainS2.sh`: Stage 2 training (`FSPS2`)
- `run_s2_only.sh`: S2-only inference

This README is written from the current scripts, config files, and directory layout in this repository.

Trained checkpoints are not included in this repository. Use your own checkpoints when running inference or evaluation.

## 1. Key Files

- `trainS1.sh`: Stage 1 training entry
- `trainS2.sh`: Stage 2 training entry
- `run_s2_only.sh`: S2-only inference entry
- `bin/FullSpectrum_Merge.py`: energy-level fusion script
- `bin/compare_metricsall.py`: metric evaluation script
- `options/train_FSPS1.yml`: Stage 1 training config
- `options/train_FSPS2.yml`: Stage 2 training config
- `test_s2_only.py`: S2-only inference script
- `test/data/val`: noisy validation data
- `test/data/cleanval/val`: clean validation labels
- `test/out/valdenoisy`: current inference output directory

## 2. Environment

Prepare a Python environment with a PyTorch build that matches your local CUDA installation, then install the repository dependencies from the project root:

```bash
pip install -r requirements.txt
python setup.py develop
```

Notes:

- The full environment for this project is defined by `requirements.txt`.
- `test_s2_only.py` reads and writes `.mat` files, so `scipy` is required and already included in `requirements.txt`.
- `python setup.py develop` installs the project in development mode.

### 2.1 Current Environment in `requirements.txt`

The current repository environment includes at least the following core packages:

- `torch==2.1.2`
- `torchvision==0.16.2`
- `basicsr==1.4.2`
- `numpy==1.24.3`
- `scipy==1.10.1`
- `scikit-image==0.21.0`
- `opencv-python==4.9.0.80`
- `pillow==10.3.0`
- `PyYAML==6.0.1`
- `einops==0.7.0`
- `pandas==2.0.3`
- `tqdm==4.66.2`
- `albumentations==0.5.2`
- `detectron2==0.6+cu113`
- `astra-toolbox==2.2.0`

The environment file also includes:

- `tensorflow==2.13.1`
- `pytorch-lightning==1.2.9`
- `tensorboard==2.13.0`
- `wandb==0.21.4`

To reproduce the same environment as this repository, use:

```bash
pip install -r requirements.txt
```

If you rebuild the environment on another machine, check compatibility for:

- `torch` and `torchvision`
- CUDA-related `nvidia-*` packages
- `detectron2`

## 3. Dataset Layout

The training configs use paired data:

- `GT / clean`: clean target images
- `LQ / noisy`: degraded or noisy input images

The current training configs use local absolute paths. Replace them with paths on your own machine before training.

Recommended directory structure:

```text
/path/to/allfentu/
  train/
  val/

/path/to/allfentu2000/
  train/
  val/
```

Requirements:

- GT and LQ filenames must match one-to-one.
- If your data is stored elsewhere, update `dataroot_gt` and `dataroot_lq` in both `options/train_FSPS1.yml` and `options/train_FSPS2.yml`.

## 4. Training

### 4.1 Stage 1 Training

`trainS1.sh` runs:

```bash
CUDA_VISIBLE_DEVICES=0 python3 FSP/train.py -opt options/train_FSPS1.yml --launcher none
```

Run it with:

```bash
sh trainS1.sh
```

Before training, check:

1. The training and validation paths in `options/train_FSPS1.yml`.
2. `CUDA_VISIBLE_DEVICES` in `trainS1.sh` if you want to use a different GPU.

Notes:

- The current script runs in single-GPU mode.
- The experiment name in `train_FSPS1.yml` is `train_FSPS1`.
- Outputs are typically saved under `experiments/train_FSPS1/`.
- `save_checkpoint_freq` is currently set to `400`.

### 4.2 Stage 2 Training

`trainS2.sh` runs:

```bash
CUDA_VISIBLE_DEVICES=0 python3 FSP/train.py -opt options/train_FSPS2.yml --launcher none
```

Run it with:

```bash
sh trainS2.sh
```

Before training, check:

1. The training and validation paths in `options/train_FSPS2.yml`.
2. `path.pretrain_network_g` points to a valid S1 checkpoint.
3. `path.pretrain_network_S1` points to the corresponding S1 checkpoint.
4. `CUDA_VISIBLE_DEVICES` in `trainS2.sh` if you want to use a different GPU.

These two fields must be set correctly:

```yaml
path:
  pretrain_network_g: /path/to/your/S1_checkpoint.pth
  pretrain_network_S1: /path/to/your/S1_checkpoint.pth
```

If you trained S1 in this repository first, those paths will usually look like:

```text
experiments/train_FSPS1/models/net_g_xxx.pth
```

Notes:

- The current script also runs in single-GPU mode.
- The experiment name in `train_FSPS2.yml` is `train_FSPS2`.
- Outputs are typically saved under `experiments/train_FSPS2/`.

## 5. S2-Only Inference

`run_s2_only.sh` runs inference directly with a trained S2 model and does not depend on S1 during testing.

Run it with:

```bash
sh run_s2_only.sh
```

The script calls:

```bash
python test_s2_only.py \
    --model_path /path/to/net_g_xxx.pth \
    --input /path/to/input \
    --output /path/to/output \
    --device cuda
```

If your model or data paths are different, update these variables in `run_s2_only.sh`:

- `MODEL_PATH`
- `INPUT_PATH`
- `OUTPUT_PATH`

Use your own S2 checkpoint path here. This repository does not provide a trained S2 model file.

### 5.1 Supported Input Formats

`test_s2_only.py` supports:

- `.png`
- `.jpg`
- `.jpeg`
- `.bmp`
- `.tiff`
- `.mat`

You can pass:

- a single image file
- a directory, in which case the script processes all supported files inside it

### 5.2 Output Format

Outputs are saved as `.mat` files with the naming pattern:

```text
original_name_processed.mat
```

Examples:

```text
84_1.mat -> 84_1_processed.mat
image.png -> image_processed.mat
```

Output details:

- variable name: `img`
- tensor layout: `CHW`
- shape meaning: `channels x height x width`

## 6. Command Examples

### 6.1 Train S1

```bash
sh trainS1.sh
```

### 6.2 Train S2

```bash
sh trainS2.sh
```

### 6.3 Batch Inference

```bash
sh run_s2_only.sh
```

### 6.4 Run Inference Without the Shell Script

```bash
python test_s2_only.py \
  --model_path /path/to/your_s2_checkpoint.pth \
  --input /path/to/input_dir \
  --output /path/to/output_dir \
  --device cuda
```

## 7. `test/` Directory Layout

The repository already includes a test directory structure you can follow:

```text
test/
  data/
    val/                 # noisy validation data
    cleanval/val/        # clean ground-truth labels
  out/
    valdenoisy/          # inference outputs
```

Current meaning of each folder:

- `test/data/val`: noisy input `.mat` files
- `test/data/cleanval/val`: matching clean label `.mat` files
- `test/out/valdenoisy`: model output files such as `xxx_processed.mat`

If you want to evaluate with the repository test layout, set the input and output paths in `run_s2_only.sh` accordingly.

## 8. Energy-Level Fusion

`bin/FullSpectrum_Merge.py` merges multi-level `.mat` outputs from the same sample into one final result.

The script currently uses hard-coded input and output paths.

```python
input_folder_path = '/path/to/multi_level_mat_files'
output_folder_path = '/path/to/fused_results'
```

Update those paths for your data, then run:

```bash
python bin/FullSpectrum_Merge.py
```

Input requirements:

- Input files must be `.mat`.
- Filenames must follow the pattern `n_1.mat` to `n_6.mat`.
- For one sample, the script expects all six files such as `200_1.mat`, `200_2.mat`, ..., `200_6.mat`.
- Each file must contain a variable named `img`.
- The current script assumes every `img` is `512 x 512`.

Output behavior:

- Six channel files are merged into one output file.
- The output filename is `n.mat`.
- Results are saved to `output_folder_path`.

## 9. Metric Evaluation

`bin/compare_metricsall.py` computes `PSNR` and `SSIM` between output results and clean labels.

The script currently compares the clean label directory and the denoised output directory. In repository-relative form, they are:

```text
clean_dir = test/data/cleanval/val
noisy_dir = test/out/valdenoisy
```

Run it with:

```bash
python bin/compare_metricsall.py
```

Matching rules:

- The script searches for matching `.mat` files in both directories.
- If an output file ends with `_processed`, the suffix is removed before matching.
- For example, `84_1.mat` is matched with `84_1_processed.mat`.

Outputs:

- Per-file, per-channel `PSNR` and `SSIM` are printed to the console.
- Average metrics are computed for each channel.
- Global average `PSNR` and `SSIM` are computed across all matched pairs.
- A log file is saved to `results/testlog/metrics.txt`.

Notes:

- The script supports both single-channel and multi-channel `.mat` files.
- Multi-channel metrics are computed channel by channel.
- GT and prediction are jointly normalized before metric computation.

## 10. Notes

- Many paths in this repository are absolute paths. Update them before running on another machine.
- Stage 2 training requires valid S1 checkpoints.
- The default scripts use `CUDA_VISIBLE_DEVICES=0`. Change that manually if needed.
- `run_s2_only.sh` is an S2-only inference path and does not run the original joint S1+S2 testing flow.
- If you use `.mat` data, make sure `scipy` is installed.
- `FullSpectrum_Merge.py` and `compare_metricsall.py` both use hard-coded absolute paths.
- `FullSpectrum_Merge.py` requires all six energy-level files and currently assumes `512 x 512` inputs.
- No trained model checkpoints are bundled in the repository.

## 11. Recommended Workflow

1. Update the dataset paths in `options/train_FSPS1.yml`.
2. Run `sh trainS1.sh`.
3. Record the S1 checkpoint path.
4. Update `pretrain_network_g` and `pretrain_network_S1` in `options/train_FSPS2.yml`.
5. Run `sh trainS2.sh`.
6. Update the model, input, and output paths in `run_s2_only.sh`.
7. Run `sh run_s2_only.sh`.
8. If needed, update and run `python bin/FullSpectrum_Merge.py`.
9. If needed, run `python bin/compare_metricsall.py`.
