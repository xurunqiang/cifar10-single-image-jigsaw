"""
Expansion strategies for v2 jigsaw puzzle reconstruction:
- BFS: Breadth-first expansion from seed (Top, Right, Bottom, Left)
- Random Frontier: Uniform random sampling among empty slots adjacent to placed slots
"""

from collections import deque
import random
from typing import List, Tuple, Set, Optional
import torch

# 4-connected directional offsets: Top, Right, Bottom, Left
DIRECTIONS = [(-1, 0), (0, 1), (1, 0), (0, -1)]
DIR_NAMES = ["Top", "Right", "Bottom", "Left"]


def get_bfs_order(grid_size: int, seed_coord: Tuple[int, int]) -> List[Tuple[int, int]]:
    """
    Breadth-first search expansion from seed_coord.
    Within each level, exploration follows Top, Right, Bottom, Left order.
    Returns: List of (r, c) coordinates of length grid_size^2 - 1.
    """
    sr, sc = seed_coord
    assert 0 <= sr < grid_size and 0 <= sc < grid_size, f"Seed {seed_coord} out of bounds"

    visited: Set[Tuple[int, int]] = {seed_coord}
    queue: deque = deque([seed_coord])
    order: List[Tuple[int, int]] = []

    while queue:
        curr_r, curr_c = queue.popleft()
        for dr, dc in DIRECTIONS:
            nr, nc = curr_r + dr, curr_c + dc
            if 0 <= nr < grid_size and 0 <= nc < grid_size:
                if (nr, nc) not in visited:
                    visited.add((nr, nc))
                    order.append((nr, nc))
                    queue.append((nr, nc))

    assert len(order) == grid_size * grid_size - 1, f"BFS order length {len(order)} != {grid_size**2 - 1}"
    return order


def get_random_frontier_order(
    grid_size: int,
    seed_coord: Tuple[int, int],
    rng: Optional[random.Random] = None
) -> List[Tuple[int, int]]:
    """
    Random frontier expansion:
    At each step, uniformly pick an empty slot that is 4-connected adjacent to
    at least one already placed slot.
    Returns: List of (r, c) coordinates of length grid_size^2 - 1.
    """
    sr, sc = seed_coord
    assert 0 <= sr < grid_size and 0 <= sc < grid_size, f"Seed {seed_coord} out of bounds"

    sampler = rng if rng is not None else random

    placed: Set[Tuple[int, int]] = {seed_coord}
    order: List[Tuple[int, int]] = []
    total_slots = grid_size * grid_size

    while len(placed) < total_slots:
        # Collect frontier (unplaced slots with at least one placed 4-neighbor)
        frontier: Set[Tuple[int, int]] = set()
        for pr, pc in placed:
            for dr, dc in DIRECTIONS:
                nr, nc = pr + dr, pc + dc
                if 0 <= nr < grid_size and 0 <= nc < grid_size and (nr, nc) not in placed:
                    frontier.add((nr, nc))

        assert len(frontier) > 0, "Frontier is unexpectedly empty!"
        # Sort first to ensure deterministic sampling when using seeded rng
        frontier_list = sorted(list(frontier))
        chosen = sampler.choice(frontier_list)
        placed.add(chosen)
        order.append(chosen)

    assert len(order) == total_slots - 1
    return order


def get_expansion_order(
    grid_size: int,
    seed_coord: Tuple[int, int],
    strategy: str = "random_frontier",
    rng: Optional[random.Random] = None
) -> List[Tuple[int, int]]:
    """
    Get expansion order based on strategy ('bfs' or 'random_frontier').
    """
    if strategy == "bfs":
        return get_bfs_order(grid_size, seed_coord)
    elif strategy == "random_frontier":
        return get_random_frontier_order(grid_size, seed_coord, rng=rng)
    else:
        raise ValueError(f"Unknown expansion strategy: {strategy}")
