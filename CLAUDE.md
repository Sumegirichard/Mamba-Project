# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

University course assignment (Deep Learning Methods for Data Analysis, Task B: Snake Recognition). It is a multitask image classifier: from one snake image, predict the **species** (296 classes after re-indexing) and whether it is **venomous** (`MIVS`, binary 0/1). The code, comments, console output and docs are in **Hungarian**, so match that when editing. Design rationale is in `docs/tervezoi_dontesek.txt`. The open TODOs are in `docs/kovetkezo_teendok.txt`. The assignment spec is the PDF in `docs/`.

There is no build system, test suite, linter or requirements file. A local `.venv` exists (TensorFlow 2.21 / Keras 3). Scripts are run directly with Python.

## Data pipeline (one-off scripts, run in this order from the repo root)

1. `src/merge.py`: left-joins `train_images_metadata.csv` with `venomous_status_metadata.csv` on `class_id` to produce `data/merged_snake_metadata.csv`.
2. `src/fix_labels.py`: uses LabelEncoder to remap the sparse original `class_id`s to dense 0..295 and writes `data/class_id_mapping.csv` (`original_class_id` → `new_model_id`). This mapping is a model artifact. You need it to decode predictions.
3. `src/create_merged_withgoodlabels.py`: applies the mapping to produce `data/final_merged_metadata_with_new_labels.csv`, the canonical dataset.
4. `src/kfold.py`: StratifiedKFold(5, seed 42) on `class_id`. It writes `fold_{k}_{train,val}.csv` into the **current directory**, and these were then moved to `data/folds/`. The older `train_split*/val_split*` CSVs are the earlier 80/20 split and are superseded by the folds.
5. `src/letterboxing.py`: converts images to RGB and pads them to a square (gray `(127,127,127)`, keeping the aspect ratio) at 224 or 384. Its paths are hardcoded to `D:/Letöltések`. Change `TARGET_SIZE`/paths before running.

The CSV columns are `observation_id, endemic, binomial_name, code, image_path, class_id, MIVS`. `image_path` is just a filename relative to the image folder. Image folders and zips are gitignored.

## Training scripts

`src/<Backbone>/` holds one self-contained script per experiment. There is no shared module, so each file duplicates the dataset, loss and metric code. The backbones are ResNet50(+iNat), InceptionV3 (iNat), EfficientNet, ConvNeXt, Swin and BioCLIP. All are TensorFlow/Keras except `BioClip/model_bioclip.py`, which uses PyTorch + `open_clip` (`hf-hub:imageomics/bioclip`). Each backbone dir typically has:
- `model_*.py`: the training script. The `*384.py` variants train at 384×384 and usually load the 224 fold weights to fine-tune further.
- `lr_finder_*.py` / `LR_finder_*backbone*.py`: LR range tests, run separately for the warm-up phase (frozen backbone) and the backbone fine-tune phase.

The top-level `src/model.py`, `src/LR_finder*.py`, `src/predict.py` and `src/mamba.py` are older ResNet50 / plain-CNN prototypes (`models/snake_multitask_model.h5`).

Run a model script for specific folds (the default is `FOLDS_TO_RUN` inside the file):
```bash
python src/ConvNext/model_convnext384.py --fold 1 2
```

**Environment:** the scripts are written primarily for **Google Colab**. When `google.colab` is importable, they mount Drive at `/content/drive/MyDrive/Mamba`, unzip `Databank/Train data/train_images_{224,384}_letterboxed.zip` to local SSD, copy folds from `CSV files/folds`, and write weights/logs/plots/JSON results to `Mamba/Logs`. The local fallback paths (`data/CSV files/folds`, `data/Databank/...`, outputs to `models/`) differ from this repo's actual layout (`data/folds/`), so check them before running locally. Hyperparameters are module-level constants at the top of each script.

## Shared modelling conventions (keep consistent across scripts)

- Two outputs: `species_head` (softmax over 296 classes) and `venomous_head` (1-unit sigmoid).
- `v_mapping` is a hardcoded 296-length 0/1 list: the venomous flag per *new* class id. It is tied to `class_id_mapping.csv` ordering, so regenerating the mapping invalidates it.
- `cost_sensitive_species_loss`: sparse CCE weighted via `v_mapping`. It penalizes predicting a non-venomous class for a venomous snake most heavily (5×).
- `weighted_binary_venomous_loss`: BCE with positive (venomous) samples weighted 5×.
- `venomous_penalty_metric`: the assignment's domain-specific metric that weights misclassifications by their venomous consequence. Macro F1 (`SparseMacroF1`) is also tracked.
- Two-phase training: warm-up with a frozen backbone, then fine-tuning at a lower LR, with EarlyStopping (restore best), ReduceLROnPlateau, a CSVLogger and per-fold `.weights.h5` checkpoints. TF scripts use `mixed_float16`. Custom losses/metrics are registered with `register_keras_serializable` so saved models reload.
- Seed 42 everywhere (`src/random_seed.py`).
- Rules from the design doc: use identical preprocessing, mapping, loss weighting and evaluation in every fold; never use the held-out test data for LR-finder or training decisions.
