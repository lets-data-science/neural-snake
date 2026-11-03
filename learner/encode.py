"""Exercise 1: turn the engine's state into the board the model reads and writes.

The model never sees the engine. It sees one 12 x 12 grid of class numbers:

    0       empty
    1       food
    2-5     head, by the direction the snake is travelling: 2 up, 3 down, 4 left, 5 right
    6-9     body, by the direction to the next segment toward the head (same order)
    10-13   tail, by the direction to the next segment toward the head (same order)

`body` lists the snake's cells as (row, col), head first and tail last. Row 0 is the top.
A one-cell snake is only a head. Food can be None (no food on the board).
"""
import numpy as np

N = 12
UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3
DIRECTION_OF_STEP = {(-1, 0): UP, (1, 0): DOWN, (0, -1): LEFT, (0, 1): RIGHT}


def encode_board(body, direction, food):
    """Return a 12 x 12 uint8 array of class numbers for this snake, its direction of travel and the food."""
    grid = np.zeros((N, N), np.uint8)
    if food is not None:
        grid[food[0], food[1]] = 1
    for i, (r, c) in enumerate(body):
        if i == 0:
            grid[r, c] = 2 + int(direction)
            continue
        toward_head = body[i - 1]
        step = DIRECTION_OF_STEP[(toward_head[0] - r, toward_head[1] - c)]
        grid[r, c] = (10 if i == len(body) - 1 else 6) + step
    return grid
