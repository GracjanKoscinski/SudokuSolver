import numpy as np
from mealpy import IntegerVar, ACOR


class SudokuACO:
    def __init__(self, initial_grid):
        self.initial_grid = initial_grid.copy()
        all_empty = list(zip(*np.where(self.initial_grid == 0)))
        all_empty.sort(
            key=lambda pos: len(self._get_candidates(pos[0], pos[1], self.initial_grid))
        )

        self.empty_pos = all_empty
        self.candidates = [
            self._get_candidates(r, c, self.initial_grid) for r, c in self.empty_pos
        ]
        self.n_vars = len(self.empty_pos)

    def _get_candidates(self, r, c, grid):
        """Znajduje dopuszczalne cyfry dla danego pola."""
        row_vals = set(grid[r, :])
        col_vals = set(grid[:, c])
        r_start, c_start = (r // 3) * 3, (c // 3) * 3
        square_vals = set(grid[r_start : r_start + 3, c_start : c_start + 3].flatten())

        forbidden = row_vals | col_vals | square_vals
        allowed = [i for i in range(1, 10) if i not in forbidden]
        return allowed if allowed else list(range(1, 10))

    def fitness_function(self, solution):
        grid = self.initial_grid.copy()
        for idx, cand_idx in enumerate(solution):
            r, c = self.empty_pos[idx]
            cand_list = self.candidates[idx]
            grid[r, c] = cand_list[int(cand_idx) % len(cand_list)]

        penalty = 0
        # Sprawdzanie wierszy i kolumn
        for i in range(9):
            row = grid[i, :]
            col = grid[:, i]
            # Używam (9 - unikalne) * mnożnik, aby mocniej różnicować wyniki
            penalty += (9 - len(np.unique(row))) ** 2
            penalty += (9 - len(np.unique(col))) ** 2

            # Kwadraty 3x3
            r_s, c_s = (i // 3) * 3, (i % 3) * 3
            subgrid = grid[r_s : r_s + 3, c_s : c_s + 3]
            penalty += (9 - len(np.unique(subgrid))) ** 2

        return float(penalty)

    def solve(self):
        if self.n_vars == 0:
            return self.initial_grid, 0.0

        lb = [0] * self.n_vars
        ub = [len(c) - 1 for c in self.candidates]

        problem_dict = {
            "bounds": IntegerVar(lb=lb, ub=ub),
            "minmax": "min",
            "obj_func": self.fitness_function,
        }

        model = ACOR.OriginalACOR(epoch=1000, pop_size=200, sample_count=50)

        termination = {
            "mode": "target",
            "quantity": 0.0,
            "epsilon": 1e-6,
            "max_early_stop": 150,
        }

        g_best = model.solve(problem_dict, termination=termination)

        final_grid = self.initial_grid.copy()
        for idx, cand_idx in enumerate(g_best.solution):
            r, c = self.empty_pos[idx]
            final_grid[r, c] = self.candidates[idx][int(cand_idx)]

        return final_grid, g_best.target


def run_aco(initial_grid, timeout=60):
    solver = SudokuACO(initial_grid)
    solution, score = solver.solve()
    return solution
