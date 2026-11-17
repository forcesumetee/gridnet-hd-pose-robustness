# How Robust Is LiDAR–Image Fusion to Camera-Pose Misalignment?

Code, notebooks and result files of the study **"How Robust Is LiDAR–Image Fusion to Camera-Pose Misalignment? A
Benchmark on GridNet-HD and a Simple Fix"** (manuscript in preparation).

A LiDAR–image fusion model reads, for every 3D point, the image pixels onto which the point projects. The projection
needs the camera poses, and poses are never exact. This repository measures what a pose error costs on
[GridNet-HD](https://huggingface.co/datasets/heig-vd-geo/GridNet-HD) (UAV LiDAR and oblique images of power-line
corridors, 11 classes), and what restores the accuracy.

- **Protocol.** The delivered camera poses are perturbed by a stated error (independent between the images, shared by
  all images of a zone, or drifting along the track), points and pixels are associated again, and *frozen* models are
  evaluated again from cached 2D outputs: per class, per error level, 10 draws of the cameras per level. Nothing is
  retrained.
- **Benchmark.** Projection voting (ImageVote), late fusion of class probabilities, feature concatenation and K×K
  window attention on a frozen LitePT-S LiDAR backbone and a frozen UPerNet (ConvNeXt-T) image network, next to the
  LiDAR-only network.
- **Fix.** Pose-noise augmentation while the small fusion head is trained, a known remedy, against attention over a
  window of neighbouring image cells.

## Results

Six validation zones, mean IoU in % with exact re-association: mean over 10 draws of the cameras and over 3 training
seeds, errors independent between the images. "Electrical" is the mean over pylon, conductor, structural cable and
insulator; σt and σr are the standard deviations, per axis, of the position error and of the rotation error of a
camera.

| Model | Pose noise in training | All classes, delivered poses | σt = 50 cm | σr = 1° | Electrical, delivered poses | σt = 50 cm | σr = 1° |
|---|---|---|---|---|---|---|---|
| LiDAR-only (LitePT-S) | – | 69.9 | 69.9 | 69.9 | 97.7 | 97.7 | 97.7 |
| ImageVote | – | 65.5 | 51.3 | 46.3 | 63.7 | 27.1 | 22.0 |
| Late fusion | no | 81.8 | 80.2 | 76.8 | 98.0 | 96.6 | 95.9 |
| Feature concatenation | no | 79.2 | 74.2 | 70.0 | 98.0 | 86.2 | 80.9 |
| Window attention 3×3 | no | 79.6 | 74.7 | 69.7 | 98.0 | 86.6 | 79.9 |
| Window attention 5×5 | no | 79.2 | 76.3 | 72.2 | 98.0 | 91.8 | 87.0 |
| Late fusion | yes | 82.0 | 81.3 | 78.7 | 98.0 | 97.8 | 97.8 |
| Feature concatenation | yes | 80.6 | 79.8 | 77.5 | 98.1 | 97.9 | 97.8 |
| Window attention 3×3 | yes | 80.6 | 79.8 | 77.3 | 98.1 | 97.9 | 97.8 |
| Window attention 5×5 | yes | 80.3 | 79.5 | 77.2 | 98.1 | 97.9 | 97.8 |

On the nine hidden test zones the final model, late fusion with pose-noise augmentation, scores 79.25 mIoU and the
LiDAR-only network 66.22 (entries `Sumetee_LitePT-S_UPerNet_LateFusion_PoseAug` and `Sumetee_LitePT-S_LiDAR-only` of
the [public leaderboard](https://huggingface.co/spaces/heig-vd-geo/GridNet-HD-Leaderboard), read on 14 November 2026).

Every table of the paper is generated from the files in `results/`: `python scripts/make_tables.py` writes
`paper/tables/*.tex`, and `python scripts/paper_numbers.py` prints the numbers that the text quotes.

## What is here

| Path | Content |
|---|---|
| `gridnet/` | the library: reading zones, the camera model, the remapping to 11 classes, voxel caches and the point–pixel association (`gridnet/preprocess/`); NumPy and SciPy only |
| `gridnet/train/` | the PyTorch side: training of the LiDAR and the image network, the cache of 2D outputs, the fusion heads, the feature-bank trainer and the robustness protocol (`protocol.py`) |
| `scripts/` | command-line entry points: caches, training, protocol, tables, figures, leaderboard files |
| `notebooks/` | the Google Colab notebooks the experiments were run with, and the two scripts that write them |
| `results/` | the result files of the runs behind the numbers of the paper |
| `paper/tables/` | the tables of the paper as `scripts/make_tables.py` writes them |
| `tests/` | unit tests on synthetic zones, and tests that tie the tables to the result files |

## What is not here

- **Data.** GridNet-HD is downloaded from Hugging Face (`heig-vd-geo/GridNet-HD`). No point and no image of the dataset
  is redistributed here, and neither are the caches built from it (the cached outputs of the image network take about
  50 MB per image).
- **Trained weights.** They are not part of this repository.
- **LitePT.** The notebooks clone the LiDAR backbone from <https://github.com/prs-eth/LitePT>.

## Installation

The library, the tables and the tests run on a CPU with Python 3.11 or newer:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest -q          # tests that need PyTorch are skipped when it is not installed
```

For the GPU side, `bash scripts/colab_env.sh <folder>` builds a self-contained Python 3.10 environment with
PyTorch 2.4.1 (CUDA 12.4), spconv, flash-attn and torch-scatter from prebuilt wheels; the notebooks call it.

## Reproducing the experiments

The experiments ran on Google Colab with Google Drive as storage. The notebooks expect a folder `GridNet/` on Drive and
the code as `GridNet/code/gridnet_code.zip`: `bash scripts/pack_code.sh` writes the archive to `data/gridnet_code.zip`,
to be uploaded there. In the order in which they depend on each other:

| Notebook | What it does |
|---|---|
| `colab_smoke_litept` | checks the environment: LitePT-S for a few hundred iterations on one zone |
| `colab_build_cache` | per-zone LiDAR caches: 5 cm voxels, labels, geometric features (CPU, high RAM) |
| `colab_train_litept` | the LiDAR-only network (LitePT-S) |
| `colab_train_upernet` | the image network (UPerNet, ConvNeXt-T) and the cache of its outputs per image |
| `colab_compact_cache2d` | packs the 2D cache into one archive per zone (CPU) |
| `colab_fusion` | ImageVote on the validation zones: association and projection voting |
| `colab_train_fusion` | fusion heads trained and evaluated on full zones |
| `colab_heads_lab` | many fusion heads side by side on stored features (the heads of the paper), with the fast protocol |
| `colab_protocol` | the robustness protocol on trained heads (exact re-association) |
| `colab_shared_error` | the protocol with errors shared by all images of a zone |
| `colab_label_transfer` | model-free label transfer under pose error (CPU) |
| `colab_qualitative` | predictions around pylons for the qualitative figure |
| `colab_leaderboard` | predictions of the test zones and the files for the leaderboard |

The explanations inside the notebooks are written in Thai. `results/experiments.csv` lists every run with its
configuration, GPU and wall time.

## The protocol

`gridnet/train/protocol.py` holds the error models (`perturbed_cams`, `shared_cams`, `drift_cams`), the seeded draws
(`draw_seed`: two runs see the same cameras for every draw they share), the fast and the exact re-association
(`rows_fast`, `rows_exact`) and the evaluation loop (`run_protocol`). `scripts/run_protocol.py` is its command line;
`scripts/plot_protocol.py`, `scripts/compare_protocol.py` and `scripts/compare_shared.py` read its output.

## Notes on the files

Comments and result notes mention `updateNN` and files under `notes/`. These are the numbered snapshots and the
journal of the development, which are not part of this repository; the column `git_commit` of
`results/experiments.csv` holds a snapshot number because the project was not under git while the experiments ran.

## Citation

A citation entry will be added when the paper is published. The dataset is GridNet-HD (Carreaud et al.,
arXiv:2601.13052).

## Acknowledgment

This work was supported by the Petchra Pra Jom Klao Master's Degree Scholarship from King Mongkut's University of
Technology Thonburi. The authors thank the creators of the GridNet-HD dataset for making the data and the public
leaderboard available.

## License

No licence has been chosen for the code yet.
