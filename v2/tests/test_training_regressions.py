"""Regression coverage for dataset isolation, stochastic evaluation and real resume."""
import random

import pytest
import torch
from torch.utils.data import DataLoader, Subset

from v2.config import ModelConfig, TrainConfig
from v2.curriculum import CurriculumScheduler, build_lr_scheduler
from v2.dataset import (JigsawProblemGenerator, build_train_val_datasets,
                        collate_jigsaw, set_dataset_epoch)
from v2.model import JigsawSolverV2
from v2.trainer import TrainerV2, compute_pairwise_accuracy


def samples():
    generator = torch.Generator().manual_seed(19)
    result = []
    for i in range(3):
        item = JigsawProblemGenerator.generate(torch.randn(3, 32, 32, generator=generator), 3,
                                               rng=random.Random(i))
        item['img_idx'] = i
        result.append(item)
    return result


def make_trainer(tmp_path, data, batch_size=2, shuffle=False, drop_last=False, device="cpu"):
    config = ModelConfig()
    train = TrainConfig(epochs=5, batch_size=batch_size, save_dir_prefix=str(tmp_path), device='cpu')
    loader = DataLoader(data, batch_size=batch_size, shuffle=shuffle,
                        drop_last=drop_last, collate_fn=collate_jigsaw)
    return TrainerV2(JigsawSolverV2(config), config, train, loader, loader, torch.device(device))


def test_disjoint_training_split_and_subset_epoch(monkeypatch):
    calls = []
    class FakeCIFAR:
        def __init__(self, root, train, download, transform):
            calls.append(train)
        def __len__(self):
            return 50
        def __getitem__(self, idx):
            return torch.arange(3072, dtype=torch.float32).reshape(3, 32, 32) / 3072, 0
    monkeypatch.setattr('v2.dataset.torchvision.datasets.CIFAR10', FakeCIFAR)
    train, val = build_train_val_datasets('/unused')
    assert calls == [True, True]
    assert len(train) == 45 and len(val) == 5
    assert set(train.indices).isdisjoint(val.indices)
    before, val_before = train[0], val[0]
    set_dataset_epoch(Subset(train, [0]), 2)
    set_dataset_epoch(val, 2)
    assert before['expansion_seed'] != train[0]['expansion_seed']
    assert not torch.equal(before['target_mapping'], train[0]['target_mapping'])
    assert torch.equal(val_before['target_mapping'], val[0]['target_mapping'])
    assert val_before['expansion_seed'] == val[0]['expansion_seed']


def test_random_frontier_validation_is_fixed(tmp_path, monkeypatch):
    trainer = make_trainer(tmp_path, samples())
    import v2.trainer as module
    original = module.solve_batch
    grids = []
    def capture(*args, **kwargs):
        assert kwargs['rng_seeds'] is not None
        result = original(*args, **kwargs)
        grids.append(result[0])
        return result
    monkeypatch.setattr(module, 'solve_batch', capture)
    first = trainer.validate()
    random.seed(9123)
    state = random.getstate()
    second = trainer.validate()
    assert random.getstate() == state
    assert first == second
    assert all(torch.equal(a, b) for a, b in zip(grids[:2], grids[2:]))


def test_pairwise_counts_translated_pairs_and_not_row_wrap():
    target = torch.arange(9).view(1, 3, 3)
    pred = torch.tensor([[[1, 2, 0], [4, 5, 3], [7, 8, 6]]])
    # Six valid vertical edges and three horizontal; zero absolute-position hits.
    assert compute_pairwise_accuracy(pred, target, 3) == pytest.approx(9 / 12)


def test_validation_weights_last_batch_by_sample_count(tmp_path, monkeypatch):
    data = samples()
    trainer = make_trainer(tmp_path, data)
    def fixed_solver(**kwargs):
        b = kwargs['patches'].shape[0]
        return torch.arange(9).view(1, 3, 3).repeat(b, 1, 1), []
    monkeypatch.setattr('v2.trainer.solve_batch', fixed_solver)
    monkeypatch.setattr('v2.trainer.compute_pairwise_accuracy', lambda pred, target, g: float(len(pred) == 2))
    assert trainer.validate()['val_pairwise_acc'] == pytest.approx(2 / 3)


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable"))])
def test_actual_epoch_resume_matches_uninterrupted_training(tmp_path, device):
    torch.manual_seed(32)
    data = samples()
    uninterrupted = make_trainer(tmp_path, data, shuffle=True, device=device)
    uninterrupted.train_epoch(1)
    uninterrupted.save_checkpoint(1, is_best=False)
    metrics = uninterrupted.train_epoch(2)
    expected = {k: v.clone() for k, v in uninterrupted.model.state_dict().items()}
    resumed = make_trainer(tmp_path, data, shuffle=True, device=device)
    resumed.load_checkpoint(str(tmp_path / 'grid3/latest.pt'))
    actual = resumed.train_epoch(2)
    assert actual == metrics
    assert all(torch.equal(expected[k], v) for k, v in resumed.model.state_dict().items())


def test_loss_average_uses_processed_samples(tmp_path):
    data = samples()
    trainer = make_trainer(tmp_path, data, drop_last=True)
    trainer.optimizer.param_groups[0]['lr'] = 0.0
    torch.manual_seed(111)
    first = trainer.train_epoch(1)['loss']
    trainer.train_loader = DataLoader(data[:2], batch_size=2, collate_fn=collate_jigsaw)
    trainer.optimizer.param_groups[0]['lr'] = 0.0
    torch.manual_seed(111)
    second = trainer.train_epoch(1)['loss']
    assert first == pytest.approx(second)


def test_curriculum_and_learning_rate_endpoints():
    curriculum = CurriculumScheduler(100)
    assert [curriculum.get_stage_info(e)['teacher_forcing_prob'] for e in (1, 20, 21, 80, 100)] == pytest.approx([1, 1, 59 / 60, 0, 0])
    assert CurriculumScheduler(2).get_stage_info(2)['teacher_forcing_prob'] == 0
    param = torch.nn.Parameter(torch.zeros(1))
    opt = torch.optim.AdamW([param], lr=3e-4)
    scheduler = build_lr_scheduler(opt, 5, 100)
    rates = []
    for _ in range(100):
        rates.append(opt.param_groups[0]['lr'])
        opt.step()
        scheduler.step()
    assert rates[0] == pytest.approx(3e-4 / 5)
    assert rates[4] == pytest.approx(3e-4)
    assert rates[-1] == pytest.approx(1e-5)


def test_real_training_controller_falls_back_when_target_is_consumed(tmp_path, monkeypatch):
    from v2.expansion import get_bfs_order
    item = samples()[0]
    order = get_bfs_order(3, item['seed_coord'])
    last_target = int(item['target_mapping'][order[-1]])

    class ForcedMistakeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.scores = torch.nn.Parameter(torch.zeros(9))
            with torch.no_grad():
                self.scores[last_target] = 10
            self.states = []
        def encode_candidates(self, patches):
            return patches.mean(dim=(-1, -2, -3)).unsqueeze(-1)
        def score_step(self, **kwargs):
            self.states.append((kwargs['grid_placed'].clone(), kwargs['used_candidates'].clone()))
            return self.scores.unsqueeze(0)

    model = ForcedMistakeModel()
    cfg = TrainConfig(epochs=100, batch_size=1, expansion_strategy='bfs', save_dir_prefix=str(tmp_path))
    loader = DataLoader([item], batch_size=1, collate_fn=collate_jigsaw)
    trainer = TrainerV2(model, ModelConfig(), cfg, loader, loader, torch.device('cpu'))
    # Inject a missed prompt at the first step, then request true placements thereafter.
    draws = iter([1.0] + [0.0] * 7)
    monkeypatch.setattr(torch, 'rand', lambda n, device=None: torch.full((n,), next(draws), device=device))
    metrics = trainer.train_epoch(1)
    assert metrics['conflict_ratio'] == pytest.approx(1 / 8)
    assert metrics['actual_prompt_ratio'] == pytest.approx(6 / 8)
    assert metrics['target_already_used_ratio'] == pytest.approx(1 / 8)
    assert torch.isfinite(torch.tensor(metrics['loss']))
    for step, (grid, used) in enumerate(model.states):
        placed = grid[grid >= 0]
        assert len(placed.unique()) == step + 1
        assert used.sum() == step + 1
        assert int(grid[0][item['seed_coord']]) == item['seed_cand']


@pytest.mark.parametrize('grid_size,mode', [(5, 'both'), (7, 'both'), (3, 'local_only'), (3, 'global_only')])
def test_grid_sizes_and_ablation_training(tmp_path, grid_size, mode):
    item = JigsawProblemGenerator.generate(torch.rand(3, 32, 32), grid_size, rng=random.Random(4))
    item['img_idx'] = 0
    config = ModelConfig(grid_size=grid_size, mode=mode)
    cfg = TrainConfig(epochs=3, batch_size=1, save_dir_prefix=str(tmp_path))
    loader = DataLoader([item], batch_size=1, collate_fn=collate_jigsaw)
    trainer = TrainerV2(JigsawSolverV2(config), config, cfg, loader, loader, torch.device('cpu'))
    metrics = trainer.train_epoch(1)
    assert torch.isfinite(torch.tensor(metrics['loss']))
    assert trainer.validate()['val_duplicate_rate'] == 0
