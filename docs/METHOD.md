# Method-to-code map

## Student-state KL

At each prefix sampled from the current student, `statewise_kl` computes either

```text
reverse: KL(student || teacher)
forward: KL(teacher || student)
```

over the full SID vocabulary. The student rollout is sampled without gradient;
gradients flow through the student distribution evaluated at the sampled
state. See `src/rethinking_opd/core.py` and `_sample_and_score_main` in
`src/rethinking_opd/grid_model.py`.

## Target anchor

For a sampled SID and its target SID, `first_divergence` locates their first
mismatched decision. The mismatch itself is already supervised on the
student-generated prefix. Target anchor therefore adds only the deeper target
prefixes that were not visited by the rollout and queries the frozen teacher on
those states. Anchor rows are selected under a fixed new-state budget, while
the complete on-policy loss remains unchanged.

The executable adapter uses forward KL on target-anchor states. The final loss
is the mean on-policy KL plus the summed anchor loss normalized by the number of
on-policy states. This is implemented in `_anchor_losses` and `training_step` in
`src/rethinking_opd/grid_model.py`.

## Public adapter

`OPDSemanticIDEncoderDecoder` subclasses GRID's semantic-ID generator. The
teacher remains frozen and outside the registered student module tree, so it is
excluded from both the optimizer and student checkpoints. The adapter reuses
the encoded user context for on-policy and target-anchor states.
