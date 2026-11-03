"""Exercise 2 (learner/food.py): the food head's input and loss. Run: python -m pytest tests/test_learner_food.py -q"""
import os
import sys

import pytest

torch = pytest.importorskip("torch")
F = torch.nn.functional
ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)
from learner.food import food_head_input, food_spawn_loss  # noqa: E402


def test_the_food_head_input_keeps_values_but_blocks_the_gradient():
    features = torch.randn(2, 33, 12, 12, requires_grad=True)
    out = food_head_input(features)
    assert torch.equal(out, features)
    assert not out.requires_grad


def batch():
    torch.manual_seed(0)
    logits = torch.randn(5, 144, requires_grad=True)
    food = torch.tensor([7, 255, 143, 30, 255], dtype=torch.uint8)
    eaten = torch.tensor([1, 1, 1, 0, 0], dtype=torch.uint8)  # rows 0 and 2 ate and spawned; row 1 ate with no spawn
    return logits, food, eaten


def test_the_loss_is_the_mean_over_rows_that_ate_and_spawned():
    logits, food, eaten = batch()
    expected = (F.cross_entropy(logits[0:1], torch.tensor([7])) + F.cross_entropy(logits[2:3], torch.tensor([143]))) / 2
    assert torch.allclose(food_spawn_loss(logits, food, eaten), expected)


def test_rows_that_did_not_spawn_get_no_gradient():
    logits, food, eaten = batch()
    food_spawn_loss(logits, food, eaten).backward()
    per_row = logits.grad.abs().sum(1)
    assert per_row[0] > 0 and per_row[2] > 0
    assert per_row[1] == 0 and per_row[3] == 0 and per_row[4] == 0


def test_a_batch_with_no_spawn_gives_zero_not_nan():
    logits = torch.randn(3, 144)
    loss = food_spawn_loss(logits, torch.full((3,), 255, dtype=torch.uint8), torch.zeros(3, dtype=torch.uint8))
    assert loss.item() == 0.0
