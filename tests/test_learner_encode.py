"""Exercise 1 (learner/encode.py): the 14-class board. Run: python -m pytest tests/test_learner_encode.py -q"""
import json
import os
import sys

import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)
from engine.snake import Snake, UP, DOWN, LEFT, RIGHT  # noqa: E402
from learner.encode import encode_board  # noqa: E402


def test_a_one_cell_snake_is_a_head_facing_its_direction():
    for direction, head_class in ((UP, 2), (DOWN, 3), (LEFT, 4), (RIGHT, 5)):
        grid = encode_board([(6, 6)], direction, (0, 0))
        assert grid.shape == (12, 12) and grid.dtype == np.uint8
        assert grid[6, 6] == head_class and grid[0, 0] == 1
        assert int((grid > 0).sum()) == 2


def test_body_and_tail_point_toward_the_head():
    # Travelling up after a turn: head (5, 6), body (6, 6) below it, tail (6, 5) to the body's left.
    grid = encode_board([(5, 6), (6, 6), (6, 5)], UP, (2, 2))
    assert grid[5, 6] == 2       # head, moving up
    assert grid[6, 6] == 6 + UP  # body: the next segment toward the head is above it
    assert grid[6, 5] == 10 + RIGHT  # tail: the next segment toward the head is to its right
    assert grid[2, 2] == 1


def test_a_two_cell_snake_has_a_head_and_a_tail_and_no_body():
    grid = encode_board([(3, 4), (3, 5)], LEFT, None)
    assert grid[3, 4] == 4 and grid[3, 5] == 10 + LEFT
    assert not ((grid >= 6) & (grid <= 9)).any()


def test_no_food_means_no_food_cell():
    grid = encode_board([(6, 6), (6, 5), (6, 4)], RIGHT, None)
    assert not (grid == 1).any()
    assert list(grid[6, 4:7]) == [10 + RIGHT, 6 + RIGHT, 5]


def test_every_state_of_45_engine_games_matches_the_recorded_board():
    traces = json.load(open(os.path.join(ROOT, "tests", "golden", "traces.json")))["traces"]
    checked = 0
    for trace in traces:
        if trace["n"] != 12:
            continue  # encode_board is the 12 x 12 board the model uses
        s = Snake(seed=trace["seed"], n=12)
        for action, state in zip(trace["actions"], trace["states"]):
            s.step(action)
            expected = np.array([int(ch, 16) for ch in state["grid14"]], np.uint8).reshape(12, 12)
            assert np.array_equal(encode_board(s.body, s.direction, s.food), expected), (trace["seed"], checked)
            checked += 1
    assert checked == 3792  # every state of the 45 twelve-by-twelve games
