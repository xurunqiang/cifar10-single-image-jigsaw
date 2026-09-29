"""
Unit tests for Curriculum and Learning Rate scheduling in v3.
"""

import torch
import pytest
from v3.curriculum import CurriculumScheduler, build_lr_scheduler


def test_curriculum_stages():
    sched = CurriculumScheduler(
        total_epochs=100,
        stage1_end=20,
        stage2_end=60,
        semantic_warmup_epochs=10,
        semantic_weight_start=0.01,
        semantic_weight_max=0.10
    )

    # Epoch 1: Stage 1 (p=1.0, lambda=0.01)
    s1 = sched.get_stage_info(1)
    assert s1["stage_name"] == "teacher_forcing"
    assert s1["tf_prob"] == 1.0
    assert abs(s1["semantic_weight"] - 0.01) < 1e-5

    # Epoch 20: Still Stage 1 (p=1.0, lambda=0.10)
    s20 = sched.get_stage_info(20)
    assert s20["stage_name"] == "teacher_forcing"
    assert s20["tf_prob"] == 1.0
    assert abs(s20["semantic_weight"] - 0.10) < 1e-5

    # Epoch 40: Middle of Stage 2 (p=0.5, lambda=0.10)
    s40 = sched.get_stage_info(40)
    assert s40["stage_name"] == "gradual_transition"
    assert abs(s40["tf_prob"] - 0.5) < 1e-5
    assert abs(s40["semantic_weight"] - 0.10) < 1e-5

    # Epoch 60: End of Stage 2 (p=0.0, lambda=0.10)
    s60 = sched.get_stage_info(60)
    assert abs(s60["tf_prob"] - 0.0) < 1e-5

    # Epoch 100: Stage 3 Autonomous (p=0.0, lambda=0.10)
    s100 = sched.get_stage_info(100)
    assert s100["stage_name"] == "autonomous"
    assert s100["tf_prob"] == 0.0


def test_lr_scheduler_warmup_and_cosine():
    param = torch.nn.Parameter(torch.zeros(1))
    opt = torch.optim.AdamW([param], lr=3e-4)
    lr_sched = build_lr_scheduler(opt, warmup_epochs=5, total_epochs=100, min_lr=1e-5)

    lrs = []
    for ep in range(100):
        lrs.append(opt.param_groups[0]["lr"])
        lr_sched.step()

    # Starts near min_lr, peaks at warmup epoch 5, decays down to min_lr
    assert lrs[0] < lrs[4]
    assert abs(lrs[4] - 3e-4) < 1e-5
    assert abs(lrs[-1] - 1e-5) < 1e-5
