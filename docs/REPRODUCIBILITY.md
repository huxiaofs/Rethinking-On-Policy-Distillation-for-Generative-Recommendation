# Public-benchmark protocol

## Data and semantic IDs

- Datasets: Amazon Beauty and Toys in the public P5/GRID format.
- Item representation: FLAN-T5-XL text embeddings.
- SID construction: three residual K-Means codebooks, width 256, trained for
  90 updates.
- Evaluation follows the same generated-SID matching convention for every
  method.

## Models and schedule

| Component | Public setting |
|---|---|
| Student | compact T5 semantic-ID generator, 13.1M parameters |
| Teacher | width-scaled model from the same family, 49M parameters |
| Student endpoint | 5k supervised updates |
| Teacher endpoint | 30k supervised updates |
| Distillation | 2k additional updates from the same student endpoint |
| SID depth / width | 3 / 256 |
| Temperature | 1.0 |
| Target-anchor fraction | 1.0 for the complete recipe |

The four main conditions share the student checkpoint, teacher checkpoint,
training data order, evaluator, and update budget. The released aggregate uses
matched runs with seeds 42, 43, and 44.

## Reproduction order

1. Run `scripts/bootstrap_grid.sh` once.
2. Run `scripts/prepare_sids.sh` for Beauty and Toys.
3. Run `scripts/train_endpoints.sh` for both datasets.
4. Run the four `METHOD` values exposed by `scripts/run_opd.sh`.
5. Evaluate each checkpoint with `scripts/evaluate.sh` and aggregate matched
   runs by dataset before computing the Beauty/Toys macro.

The scripts intentionally expose all local paths through `config/local.env` or
environment variables. They contain no cluster hostnames, user accounts, or
private storage layout.
