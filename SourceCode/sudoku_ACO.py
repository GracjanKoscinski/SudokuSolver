import numpy as np
from aco import AntColony
import matplotlib.pyplot as plt
import time

def evaluate(solution):
    score = 0
    for i in range(9):
        score += len(set(solution[i]))  # Row score
        score += len(set(solution[:, i]))  # Column score
    for i in range(0, 9, 3):
        for j in range(0, 9, 3):
            score += len(set(solution[i:i + 3, j:j + 3].flatten()))  # Subgrid score
    return score

# Define the Sudoku problem
# easy
problem = np.array([
    [9, 0, 2, 0, 7, 8, 4, 0, 0],
    [1, 8, 5, 0, 0, 0, 7, 6, 0],
    [0, 7, 0, 5, 0, 0, 0, 0, 0],
    [7, 5, 0, 0, 0, 6, 0, 8, 4],
    [4, 0, 6, 0, 0, 0, 1, 5, 7],
    [0, 1, 0, 0, 4, 5, 3, 9, 6],
    [0, 2, 0, 0, 0, 0, 5, 0, 9],
    [6, 0, 9, 0, 0, 7, 8, 3, 0],
    [0, 0, 0, 9, 8, 4, 0, 7, 0]
], dtype=int)

correct_solution = np.array([
    [9, 6, 2, 3, 7, 8, 4, 1, 5],
    [1, 8, 5, 4, 2, 9, 7, 6, 3],
    [3, 7, 4, 5, 6, 1, 9, 2, 8],
    [7, 5, 3, 1, 9, 6, 2, 8, 4],
    [4, 9, 6, 8, 3, 2, 1, 5, 7],
    [2, 1, 8, 7, 4, 5, 3, 9, 6],
    [8, 2, 7, 6, 1, 3, 5, 4, 9],
    [6, 4, 9, 2, 5, 7, 8, 3, 1],
    [5, 3, 1, 9, 8, 4, 6, 7, 2]
], dtype=int)
fixed_positions = problem != 0
# medium
# problem = np.array([
#     [9, 6, 0, 0, 0, 0, 4, 0, 0],
#     [1, 8, 5, 4, 2, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0, 1, 9, 0, 8],
#     [5, 3, 0, 9, 8, 0, 6, 0, 0],
#     [0, 4, 9, 0, 0, 0, 0, 0, 1],
#     [8, 2, 7, 0, 0, 0, 0, 4, 9],
#     [7, 5, 0, 1, 0, 6, 0, 8, 0],
#     [0, 0, 6, 8, 3, 0, 0, 5, 0],
#     [0, 1, 8, 7, 0, 0, 3, 0, 6]], dtype=int)
#
# correct_solution = np.array([
#     [9, 6, 2, 3, 7, 8, 4, 1, 5],
#     [1, 8, 5, 4, 2, 9, 7, 6, 3],
#     [3, 7, 4, 5, 6, 1, 9, 2, 8],
#     [5, 3, 1, 9, 8, 4, 6, 7, 2],
#     [6, 4, 9, 2, 5, 7, 8, 3, 1],
#     [8, 2, 7, 6, 1, 3, 5, 4, 9],
#     [7, 5, 3, 1, 9, 6, 2, 8, 4],
#     [4, 9, 6, 8, 3, 2, 1, 5, 7],
#     [2, 1, 8, 7, 4, 5, 3, 9, 6]
# ], dtype=int)
#
# fixed_positions = problem != 0

# hard
# problem = np.array([
#     [5, 0, 6, 1, 0, 2, 0, 0, 0],
#     [0, 0, 0, 6, 5, 0, 0, 1, 7],
#     [8, 1, 0, 0, 0, 0, 0, 5, 0],
#     [0, 0, 0, 2, 0, 0, 0, 9, 0],
#     [9, 0, 0, 5, 0, 7, 0, 0, 8],
#     [0, 5, 1, 0, 3, 9, 0, 4, 0],
#     [0, 0, 8, 0, 0, 0, 4, 0, 9],
#     [7, 6, 5, 9, 8, 4, 0, 0, 0],
#     [0, 0, 9, 0, 0, 0, 0, 0, 0]], dtype=int)
#
# correct_solution = np.array([
#     [5, 7, 6, 1, 9, 2, 3, 8, 4],
#     [4, 9, 3, 6, 5, 8, 2, 1, 7],
#     [8, 1, 2, 4, 7, 3, 9, 5, 6],
#     [3, 8, 7, 2, 4, 6, 5, 9, 1],
#     [9, 2, 4, 5, 1, 7, 6, 3, 8],
#     [6, 5, 1, 8, 3, 9, 7, 4, 2],
#     [1, 3, 8, 7, 2, 5, 4, 6, 9],
#     [7, 6, 5, 9, 8, 4, 1, 2, 3],
#     [2, 4, 9, 3, 6, 1, 8, 7, 5]
# ], dtype=int)
# fixed_positions = problem != 0
class SudokuAntColony(AntColony):
    def __init__(self, problem, ant_count=50, iterations=100, alpha=1.0, beta=1.0, rho=0.5, Q=1.0):
        self.problem = problem
        self.nodes = [(i, j) for i in range(9) for j in range(9) if problem[i, j] == 0]
        self.pheromone = np.ones((9, 9, 9))
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.Q = Q
        super().__init__(self.nodes, ant_count=ant_count, iterations=iterations, alpha=alpha, beta=beta, pheromone_evaporation_rate=rho, pheromone_constant=Q)

    def _get_distance(self, start, end):
        return 1  # All nodes are equally distant

    def _update_pheromone(self, ant):
        solution_matrix = self.problem.copy()
        for (i, j), num in zip(self.nodes, ant.path):
            solution_matrix[i, j] = num + 1
        score = evaluate(solution_matrix)
        for (i, j), num in zip(self.nodes, ant.path):
            self.pheromone[i, j, num] += self.Q / score

    def _get_probabilities(self, ant, index):
        (i, j) = self.nodes[index]
        valid_numbers = [num for num in range(9) if self.is_valid_move(ant, i, j, num + 1)]
        pheromones = np.array([self.pheromone[i, j, num] for num in valid_numbers])
        heuristics = np.ones(len(valid_numbers))
        probabilities = (pheromones ** self.alpha) * (heuristics ** self.beta)
        return probabilities / probabilities.sum(), valid_numbers

    def is_valid_move(self, ant, i, j, num):
        solution_matrix = self.problem.copy()
        for (r, c), n in zip(self.nodes[:len(ant.path)], ant.path):
            solution_matrix[r, c] = n + 1
        if num in solution_matrix[i, :] or num in solution_matrix[:, j]:
            return False
        box_start_row, box_start_col = 3 * (i // 3), 3 * (j // 3)
        if num in solution_matrix[box_start_row:box_start_row + 3, box_start_col:box_start_col + 3]:
            return False
        return True

    def get_path(self):
        best_path = None
        best_score = -1
        for _ in range(self.iterations):
            self.iteration()
            for ant in self.ants:
                solution_matrix = self.problem.copy()
                for (i, j), num in zip(self.nodes, ant.path):
                    solution_matrix[i, j] = num + 1
                score = evaluate(solution_matrix)
                if score > best_score:
                    best_score = score
                    best_path = ant.path
        return best_path

    def iteration(self):
        self.ants = [self.create_ant() for _ in range(self.ant_count)]
        for ant in self.ants:
            for index in range(len(self.nodes)):
                probabilities, valid_numbers = self._get_probabilities(ant, index)
                if len(valid_numbers) == 0:
                    break
                chosen_number = np.random.choice(valid_numbers, p=probabilities)
                ant.path.append(chosen_number)
            self._update_pheromone(ant)

    def create_ant(self):
        class Ant:
            def __init__(self):
                self.path = []

        return Ant()


# Solve Sudoku
start_time = time.time()
colony = SudokuAntColony(problem, ant_count=100, iterations=500, alpha=1.0, beta=1.0, rho=0.5, Q=1.0)
best_path = colony.get_path()
end_time = time.time()
print("Time taken: {0:.2f} seconds".format(end_time - start_time))
# Convert the best path to the final solution
solution = problem.copy()
for (i, j), num in zip(colony.nodes, best_path):
    solution[i, j] = num + 1

def plot_sudoku_pair(initial_sudoku, solution, correct_solution, fixed_positions):
    fig, axs = plt.subplots(1, 2, figsize=(20, 10))
    titles = ["Initial Sudoku Puzzle", "Computed Solution"]
    sudokus = [initial_sudoku, solution]

    for ax, sudoku, title in zip(axs, sudokus, titles):
        ax.set_title(title)
        ax.axis('off')

        # Draw the Sudoku grid
        for i in range(10):
            if i % 3 == 0:
                ax.plot([0, 9], [i, i], color='black', linewidth=2)
                ax.plot([i, i], [0, 9], color='black', linewidth=2)
            else:
                ax.plot([0, 9], [i, i], color='black', linewidth=1)
                ax.plot([i, i], [0, 9], color='black', linewidth=1)

        # Fill the grid with numbers
        for i in range(9):
            for j in range(9):
                if sudoku[i, j] != 0:
                    if fixed_positions[i, j]:
                        ax.text(j + 0.5, i + 0.5, sudoku[i, j], va='center', ha='center', color='black', fontsize=16)
                    else:
                        color = 'blue'
                        if title == "Computed Solution":
                            if sudoku[i, j] == correct_solution[i, j]:
                                color = 'green'
                            else:
                                color = 'red'
                        ax.text(j + 0.5, i + 0.5, sudoku[i, j], va='center', ha='center', color=color, fontsize=16)

        ax.invert_yaxis()

    plt.savefig('sudoku_ACO_easy_extended.png')
    plt.show()


print("Solved Sudoku:")
print(solution)
plot_sudoku_pair(problem, solution, correct_solution, fixed_positions)
