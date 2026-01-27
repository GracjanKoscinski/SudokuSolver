import numpy as np
import pygad
import warnings

warnings.filterwarnings("ignore")


class SudokuGASolver:

    def __init__(self, initial_sudoku, correct_solution=None):

        self.initial_sudoku = initial_sudoku
        self.correct_solution = correct_solution
        self.fixed_positions = initial_sudoku != 0

        self.stagnation_counter = 0
        self.best_fitness_history = []
        self.max_stagnation = 200  # Próg migracji
        self.found_solution = None

    def create_initial_population(self, size):

        population = []
        for _ in range(size):
            individual = self.initial_sudoku.copy()
            for i in range(0, 9, 3):
                for j in range(0, 9, 3):
                    subgrid = individual[i : i + 3, j : j + 3].flatten()
                    missing = [num for num in range(1, 10) if num not in subgrid]
                    np.random.shuffle(missing)
                    subgrid[subgrid == 0] = missing
                    individual[i : i + 3, j : j + 3] = subgrid.reshape((3, 3))
            population.append(individual.flatten())
        return np.array(population)

    def fitness_func(self, ga_instance, solution, solution_idx):
        """Fitness: suma unikalnych cyfr w rzędach i kolumnach (Max: 162)."""
        grid = solution.reshape((9, 9)).astype(int)
        score = 0
        for i in range(9):
            score += len(np.unique(grid[i, :]))  # Unikalne w wierszu
            score += len(np.unique(grid[:, i]))  # Unikalne w kolumnie
        return float(score)

    def local_search(self, individual):
        """Iterative Hill Climbing"""

        grid = individual.reshape((9, 9)).astype(int)
        best_fitness = self.fitness_func(None, individual, None)
        while True:
            improved = False
            # Losowa kolejność bloków
            blocks = np.arange(9)
            np.random.shuffle(blocks)

            for block_id in blocks:
                r, c = (block_id // 3) * 3, (block_id % 3) * 3
                fixed_block = self.fixed_positions[r : r + 3, c : c + 3].flatten()
                mutable_indices = np.where(~fixed_block)[0]
                if len(mutable_indices) < 2:
                    continue
                found_better = False
                for i in range(len(mutable_indices)):
                    for j in range(i + 1, len(mutable_indices)):
                        idx1, idx2 = mutable_indices[i], mutable_indices[j]
                        # Kopia fragmentu
                        flat_sub = grid[r : r + 3, c : c + 3].flatten()
                        # Swap
                        flat_sub[idx1], flat_sub[idx2] = flat_sub[idx2], flat_sub[idx1]
                        test_grid = grid.copy()
                        test_grid[r : r + 3, c : c + 3] = flat_sub.reshape(3, 3)
                        new_fitness = self.fitness_func(None, test_grid.flatten(), None)
                        if new_fitness > best_fitness:
                            grid = test_grid
                            best_fitness = new_fitness
                            improved = True
                            found_better = True
                            break
                    if found_better:
                        break
                if found_better:
                    break
            if not improved:
                break
        return grid.flatten()

    def custom_crossover(self, parents, offspring_size, ga_instance):

        offspring = []

        for k in range(offspring_size[0]):
            p1 = parents[k % parents.shape[0]].reshape(9, 9)
            p2 = parents[(k + 1) % parents.shape[0]].reshape(9, 9)
            child = np.copy(p1)
            for b in range(9):  # Crossover na poziomie bloków 3x3
                if np.random.rand() < 0.5:
                    r, c = (b // 3) * 3, (b % 3) * 3
                    child[r : r + 3, c : c + 3] = p2[r : r + 3, c : c + 3]
            offspring.append(child.flatten())
        return np.array(offspring)

    def custom_mutation(self, offspring, ga_instance):

        # Adaptacyjna szansa na mutację
        rate = 0.4 if self.stagnation_counter > 50 else 0.1
        for idx in range(offspring.shape[0]):
            if np.random.rand() < rate:
                grid = offspring[idx].reshape(9, 9)
                b = np.random.randint(0, 9)
                r, c = (b // 3) * 3, (b % 3) * 3
                fixed_sub = self.fixed_positions[r : r + 3, c : c + 3].flatten()
                m_idx = np.where(~fixed_sub)[0]
                if len(m_idx) >= 2:
                    i1, i2 = np.random.choice(m_idx, 2, replace=False)
                    flat = grid[r : r + 3, c : c + 3].flatten()
                    flat[i1], flat[i2] = flat[i2], flat[i1]
                    grid[r : r + 3, c : c + 3] = flat.reshape(3, 3)
                offspring[idx] = grid.flatten()
        return offspring

    def on_generation(self, ga_instance):

        best_sol, best_fit, best_idx = ga_instance.best_solution()
        if best_fit >= 150:
            improved = self.local_search(best_sol)
            improved_fit = self.fitness_func(None, improved, None)

            if (
                improved_fit == best_fit
                and self.stagnation_counter > 15
                and best_fit >= 158
            ):
                improved = self.custom_mutation(improved.reshape(1, -1), ga_instance)[0]
                improved_fit = self.fitness_func(None, improved, None)
                self.stagnation_counter = 0

            ga_instance.population[best_idx] = improved
            best_sol = improved
            best_fit = improved_fit

        self.best_fitness_history.append(best_fit)
        if ga_instance.generations_completed % 100 == 0:
            print(
                f"Gen {ga_instance.generations_completed}: Fitness {best_fit:.0f}/162 (Stagnacja: {self.stagnation_counter})"
            )

        if best_fit == 162:
            print(f"!!! Rozwiązano w pokoleniu {ga_instance.generations_completed} !!!")
            self.found_solution = best_sol.reshape(9, 9)
            return "stop"

        if (
            len(self.best_fitness_history) > 1
            and best_fit <= self.best_fitness_history[-2]
        ):
            self.stagnation_counter += 1
        else:
            self.stagnation_counter = 0

        if self.stagnation_counter >= self.max_stagnation:
            print(" -> Wykryto stagnację. Migracja nowych osobników (Island Model)...")
            # Wymień 80% najsłabszych na nowe losowe rozwiązania
            num_new = int(ga_instance.sol_per_pop * 0.8)
            new_pop = self.create_initial_population(num_new)
            indices = np.argsort(ga_instance.last_generation_fitness)[:num_new]
            ga_instance.population[indices] = new_pop
            self.stagnation_counter = 0


def run_ga(
    initial_sudoku, correct_solution=None, population_size=1000, num_generations=1000
):
    solver = SudokuGASolver(initial_sudoku, correct_solution)
    ga_instance = pygad.GA(
        num_generations=num_generations,
        num_parents_mating=population_size // 4,
        fitness_func=solver.fitness_func,
        sol_per_pop=population_size,
        num_genes=81,
        initial_population=solver.create_initial_population(population_size),
        parent_selection_type="tournament",
        K_tournament=5,
        crossover_type=solver.custom_crossover,
        mutation_type=solver.custom_mutation,
        on_generation=solver.on_generation,
        keep_elitism=5,
        suppress_warnings=True,
    )
    ga_instance.run()
    final = (
        solver.found_solution
        if solver.found_solution is not None
        else ga_instance.best_solution()[0].reshape(9, 9)
    )
    return final.astype(int)
