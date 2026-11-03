"""Exercise 2: teach the food head without letting it damage the rest of the model.

When the snake eats, new food appears on a random empty cell. No model can predict a random
cell, so the best possible food prediction is "every empty cell equally likely". In Run 1 the
food head tried anyway: it memorised the training spawns, and its large gradient flowed back
into the layers it shares with the board prediction. One-step accuracy fell from 97.25% at
5 minutes to 75.8% at 60 minutes. Run 2 cut that path, and reached 99.98%.

You write the two pieces that fix it.
"""
import torch
import torch.nn.functional as F

NO_SPAWN = 255  # the food value stored on rows where no new food appeared
CELLS = 144     # 12 x 12


def food_head_input(features):
    """The shared features the food head reads, with no gradient flowing back into them."""
    return features.detach()


def food_spawn_loss(food_logits, food, eaten):
    """Mean cross-entropy of the food head, over the rows where the snake ate and new food appeared.

    food_logits: float tensor [batch, 144], one score per cell.
    food:        uint8 tensor [batch], the cell index where food appeared, or 255 when none did.
    eaten:       uint8 tensor [batch], 1 when the snake ate on this step.

    Rows where nothing spawned contribute nothing. A batch with no such rows gives a loss of 0.
    """
    valid = ((eaten == 1) & (food < CELLS)).float()
    target = food.long().clamp(max=CELLS - 1)  # any in-range index works on masked rows; their loss is multiplied by 0
    ce = F.cross_entropy(food_logits, target, reduction="none")
    return (ce * valid).sum() / valid.sum().clamp(min=1.0)
