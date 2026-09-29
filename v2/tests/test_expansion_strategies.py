"""
Test 3: Expansion Strategies Validation
Verify:
- Both 'bfs' and 'random_frontier' visit all non-seed slots exactly once.
- Random frontier expansion is strictly 4-adjacent to the currently placed region at every step.
- Connectivity is maintained throughout.
"""

import pytest
import random
from v2.expansion import get_bfs_order, get_random_frontier_order, DIRECTIONS


@pytest.mark.parametrize("grid_size", [3, 5, 7])
def test_expansion_coverage_and_connectivity(grid_size):
    # Test multiple seed positions: corners, edges, center
    seeds = [
        (0, 0),
        (0, grid_size - 1),
        (grid_size // 2, grid_size // 2),
        (grid_size - 1, grid_size - 1),
    ]

    for sr, sc in seeds:
        seed_coord = (sr, sc)
        total_slots = grid_size * grid_size

        # --- Test BFS ---
        bfs_order = get_bfs_order(grid_size, seed_coord)
        assert len(bfs_order) == total_slots - 1
        assert seed_coord not in bfs_order
        assert len(set(bfs_order)) == total_slots - 1

        # Check BFS connectivity
        placed = {seed_coord}
        for r, c in bfs_order:
            assert 0 <= r < grid_size and 0 <= c < grid_size
            has_placed_neighbor = any((r + dr, c + dc) in placed for dr, dc in DIRECTIONS)
            assert has_placed_neighbor, f"BFS slot {(r, c)} was not connected to placed region!"
            placed.add((r, c))
        assert len(placed) == total_slots

        # --- Test Random Frontier ---
        rng = random.Random(12345)
        for _ in range(5):  # test 5 random runs
            rf_order = get_random_frontier_order(grid_size, seed_coord, rng=rng)
            assert len(rf_order) == total_slots - 1
            assert seed_coord not in rf_order
            assert len(set(rf_order)) == total_slots - 1

            placed = {seed_coord}
            for step_idx, (r, c) in enumerate(rf_order):
                assert 0 <= r < grid_size and 0 <= c < grid_size
                has_placed_neighbor = any((r + dr, c + dc) in placed for dr, dc in DIRECTIONS)
                assert has_placed_neighbor, (
                    f"Step {step_idx}: slot {(r, c)} was not 4-adjacent to placed region!"
                )
                placed.add((r, c))
            assert len(placed) == total_slots
