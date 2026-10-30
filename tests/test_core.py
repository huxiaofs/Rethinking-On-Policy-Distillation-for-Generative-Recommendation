import torch

from rethinking_opd.core import first_divergence, select_target_anchor_rows, statewise_kl


def test_kl_directions_match_torch_reference():
    student = torch.tensor([[0.2, -0.5, 1.1]], dtype=torch.float64)
    teacher = torch.tensor([[-0.1, 0.7, 0.3]], dtype=torch.float64)
    student_logp = torch.log_softmax(student, dim=-1)
    teacher_logp = torch.log_softmax(teacher, dim=-1)
    expected_reverse = torch.sum(student_logp.exp() * (student_logp - teacher_logp), -1)
    expected_forward = torch.sum(teacher_logp.exp() * (teacher_logp - student_logp), -1)
    torch.testing.assert_close(
        statewise_kl(student, teacher, "reverse"), expected_reverse.float()
    )
    torch.testing.assert_close(
        statewise_kl(student, teacher, "forward"), expected_forward.float()
    )


def test_anchor_selects_only_genuinely_new_states_with_budget():
    sampled = torch.tensor(
        [[1, 1, 1], [1, 2, 2], [3, 3, 3], [4, 4, 4], [5, 5, 5]]
    )
    target = torch.tensor(
        [[1, 9, 9], [1, 2, 9], [8, 8, 8], [4, 4, 4], [7, 7, 7]]
    )
    # Row 0 costs one state (fork lv1), rows 2/4 cost two; row 1 first
    # diverges at the final action and therefore adds no unseen prefix state.
    rows, forks = select_target_anchor_rows(
        sampled, target, state_budget=3, generator=torch.Generator().manual_seed(7)
    )
    assert 0 in rows.tolist()
    assert 1 not in rows.tolist()
    assert 3 not in rows.tolist()
    assert int((2 - forks).sum()) <= 3
    assert set(forks.tolist()).issubset({0, 1})


def test_anchor_empty_when_only_final_action_differs():
    sampled = torch.tensor([[1, 2, 3], [4, 5, 6]])
    target = torch.tensor([[1, 2, 9], [4, 5, 8]])
    rows, forks = select_target_anchor_rows(sampled, target, state_budget=2)
    assert rows.numel() == 0
    assert forks.numel() == 0


def test_first_divergence_supports_general_sid_depth():
    sampled = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8], [9, 1, 2, 3]])
    target = torch.tensor([[1, 2, 0, 4], [5, 6, 7, 8], [0, 1, 2, 3]])
    assert first_divergence(sampled, target).tolist() == [2, 4, 0]
