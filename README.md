# When Stronger Teachers Yield Weaker Students

Code for **“When Stronger Teachers Yield Weaker Students: Rethinking
On-Policy Distillation for Generative Recommendation.”** The release implements
full-vocabulary forward/reverse KL on student-generated SID states and the
target-anchor state supplementation used in the paper. It also provides the
complete public reproduction on Amazon Beauty and Toys.

## Setup

```bash
git clone <repository-url>
cd rethinking-opd-gr

# Fetch the exact GRID revision used by the public experiments.
bash scripts/bootstrap_grid.sh

python -m venv .venv
source .venv/bin/activate
pip install -r third_party/GRID/requirements.txt
pip install -e '.[test]'
```

The bootstrap script pins GRID to commit
`2fe3475b2d369580234093f35d52b1a2f54d0472` and preserves its upstream license
and attribution files.

## Layout

```text
config/
  paths.env.example          local path template and GRID experiment overrides
scripts/
  bootstrap_grid.sh          fetch the pinned public backbone
  prepare_sids.sh            embeddings -> RQ-KMeans -> three-level SIDs
  train_endpoints.sh         train the 13.1M student and 49M teacher
  run_opd.sh                 reverse/forward KL, with or without target anchor
  evaluate.sh                evaluate a checkpoint with the GRID evaluator
src/rethinking_opd/
  core.py                    KL objectives and target-anchor selection
  grid_model.py              executable GRID training adapter
  grid_train.py              Hydra training entry point
  grid_validate.py           standalone validation entry point
tests/                       framework-independent unit tests
results/                     released aggregate public-benchmark results
```

Generated data, checkpoints, logs, and `output/` are ignored by Git.

## Data

We use the P5-formatted Amazon Beauty and Toys splits released by GRID. Download
the [preprocessed Amazon data](https://drive.google.com/file/d/1B5_q_MT3GYxmHLrMK0-lAqgpbAuikKEz/view?usp=sharing)
and keep the following structure:

```text
amazon_data/
  beauty/{training,evaluation,testing,items}/
  toys/{training,evaluation,testing,items}/
```

Copy the path template and edit it for the local machine:

```bash
cp config/paths.env.example config/local.env
```

## Usage

The commands below reproduce one dataset at a time. Set `DATASET=beauty` or
`DATASET=toys`; all artifacts are written under `OUTPUT_ROOT`.

### 1. Build semantic IDs

```bash
DATASET=beauty bash scripts/prepare_sids.sh
```

This generates FLAN-T5-XL item embeddings and learns three RQ-KMeans codebooks
of width 256.

### 2. Train student and teacher endpoints

```bash
DATASET=beauty bash scripts/train_endpoints.sh
```

The public protocol uses a 5k-update 13.1M student checkpoint and a 30k-update
49M teacher checkpoint. Set the resulting paths before distillation:

```bash
export STUDENT_CHECKPOINT=/path/to/student-5k.ckpt
export TEACHER_CHECKPOINT=/path/to/teacher-30k.ckpt
```

### 3. Run the OPD comparison

```bash
DATASET=beauty METHOD=reverse        bash scripts/run_opd.sh
DATASET=beauty METHOD=forward        bash scripts/run_opd.sh
DATASET=beauty METHOD=reverse_anchor bash scripts/run_opd.sh
DATASET=beauty METHOD=forward_anchor bash scripts/run_opd.sh
```

`MAX_STEPS=7000` resumes the 5k student and performs the paper's 2k-update
distillation budget. `ANCHOR_FRACTION` controls the target-anchor state budget
and defaults to `1.0`.

### 4. Evaluate

```bash
DATASET=beauty \
CHECKPOINT=/path/to/distilled.ckpt \
RUN_NAME=forward_anchor \
bash scripts/evaluate.sh
```

## Expected public results

Three matched runs produce the following Beauty/Toys macro recall (mean ± SD):

| Method | R@5 | R@10 |
|---|---:|---:|
| reverse KL | 5.341 ± 0.153 | 7.792 ± 0.060 |
| reverse KL + target anchor | 5.573 ± 0.119 | 8.130 ± 0.111 |
| forward KL | 5.641 ± 0.082 | 8.219 ± 0.179 |
| forward KL + target anchor | **5.661 ± 0.036** | **8.312 ± 0.101** |

The machine-readable values are in `results/public_macro.csv`. Industrial data
and model checkpoints are not part of this public artifact.

## Tests

```bash
pytest
```

See `docs/METHOD.md` for the implementation mapping and
`docs/REPRODUCIBILITY.md` for the released experimental protocol.

## License

The code in this repository is released under the Apache License 2.0. GRID is
fetched separately and remains under its upstream non-commercial research
license.
