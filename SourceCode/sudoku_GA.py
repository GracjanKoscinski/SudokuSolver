import time
import numpy as np
import pygad
import matplotlib.pyplot as plt
# easy
# initial_sudoku = np.array([
#     [9, 0, 2, 0, 7, 8, 4, 0, 0],
#     [1, 8, 5, 0, 0, 0, 7, 6, 0],
#     [0, 7, 0, 5, 0, 0, 0, 0, 0],
#     [7, 5, 0, 0, 0, 6, 0, 8, 4],
#     [4, 0, 6, 0, 0, 0, 1, 5, 7],
#     [0, 1, 0, 0, 4, 5, 3, 9, 6],
#     [0, 2, 0, 0, 0, 0, 5, 0, 9],
#     [6, 0, 9, 0, 0, 7, 8, 3, 0],
#     [0, 0, 0, 9, 8, 4, 0, 7, 0]
# ], dtype=int)
#
# initial_sudoku_solution = np.array([
#     [9, 6, 2, 3, 7, 8, 4, 1, 5],
#     [1, 8, 5, 4, 2, 9, 7, 6, 3],
#     [3, 7, 4, 5, 6, 1, 9, 2, 8],
#     [7, 5, 3, 1, 9, 6, 2, 8, 4],
#     [4, 9, 6, 8, 3, 2, 1, 5, 7],
#     [2, 1, 8, 7, 4, 5, 3, 9, 6],
#     [8, 2, 7, 6, 1, 3, 5, 4, 9],
#     [6, 4, 9, 2, 5, 7, 8, 3, 1],
#     [5, 3, 1, 9, 8, 4, 6, 7, 2]
# ], dtype=int)
# fixed_positions = initial_sudoku != 0
#
#
#  medium
# initial_sudoku = np.array([
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
# initial_sudoku_solution = np.array([
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
# fixed_positions = initial_sudoku != 0

# hard
initial_sudoku = np.array([
    [5, 0, 6, 1, 0, 2, 0, 0, 0],
    [0, 0, 0, 6, 5, 0, 0, 1, 7],
    [8, 1, 0, 0, 0, 0, 0, 5, 0],
    [0, 0, 0, 2, 0, 0, 0, 9, 0],
    [9, 0, 0, 5, 0, 7, 0, 0, 8],
    [0, 5, 1, 0, 3, 9, 0, 4, 0],
    [0, 0, 8, 0, 0, 0, 4, 0, 9],
    [7, 6, 5, 9, 8, 4, 0, 0, 0],
    [0, 0, 9, 0, 0, 0, 0, 0, 0]], dtype=int)

initial_sudoku_solution = np.array([
    [5, 7, 6, 1, 9, 2, 3, 8, 4],
    [4, 9, 3, 6, 5, 8, 2, 1, 7],
    [8, 1, 2, 4, 7, 3, 9, 5, 6],
    [3, 8, 7, 2, 4, 6, 5, 9, 1],
    [9, 2, 4, 5, 1, 7, 6, 3, 8],
    [6, 5, 1, 8, 3, 9, 7, 4, 2],
    [1, 3, 8, 7, 2, 5, 4, 6, 9],
    [7, 6, 5, 9, 8, 4, 1, 2, 3],
    [2, 4, 9, 3, 6, 1, 8, 7, 5]
], dtype=int)
fixed_positions = initial_sudoku != 0
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

    plt.savefig('sudoku_ga_easy.png')
    plt.show()
def create_initial_population(size):
    population = []
    for _ in range(size):
        individual = initial_sudoku.copy()
        for i in range(0, 9, 3):
            for j in range(0, 9, 3):
                subgrid = individual[i:i + 3, j:j + 3].flatten()
                missing_numbers = [num for num in range(1, 10) if num not in subgrid]
                np.random.shuffle(missing_numbers)
                subgrid[subgrid == 0] = missing_numbers
                individual[i:i + 3, j:j + 3] = subgrid.reshape((3, 3))
        population.append(individual.flatten())
    return np.array(population)


# Generate initial population
population_size = 50
initial_population = create_initial_population(population_size)


def on_generation(ga_instance):
    best_solution, best_fitness, best_idx = ga_instance.best_solution()
    print(f"Generation {ga_instance.generations_completed}: Best fitness = {best_fitness}")

    if best_fitness == 81 or ga_instance.generations_completed == ga_instance.num_generations - 1:
        print("Stopping GA as best fitness = 81")
        print("Best solution:")
        print(best_solution.reshape((9, 9)).astype(int), best_fitness)
        plot_sudoku_pair(initial_sudoku, best_solution.reshape((9, 9)).astype(int), initial_sudoku_solution, fixed_positions)
        return 'stop'
    else:
        print("Continuing GA as best fitness != 81")


def fitness_func(ga_instance, solution, solution_idx):
    solution = solution.reshape((9, 9))
    score = 0

    # Penalize repeated numbers in rows and columns
    for i in range(9):
        row_counts = np.bincount(solution[i, :].astype(int))  # Konwersja na int przed użyciem np.bincount
        col_counts = np.bincount(solution[:, i].astype(int))  # Konwersja na int przed użyciem np.bincount
        score -= (np.count_nonzero(row_counts > 1) + np.count_nonzero(col_counts > 1))

    # Check 3x3 subgrids
    for i in range(0, 9, 3):
        for j in range(0, 9, 3):
            subgrid = solution[i:i + 3, j:j + 3].flatten()
            score += len(set(subgrid))

    return score




def custom_crossover(parents, offspring_size, ga_instance):
    offspring = []
    for k in range(offspring_size[0]):
        parent1_idx = k % parents.shape[0]
        parent2_idx = (k + 1) % parents.shape[0]
        parent1 = parents[parent1_idx].reshape(9, 9)
        parent2 = parents[parent2_idx].reshape(9, 9)

        # Choose random grids from parent1 (mother)
        m = np.random.randint(1, 9)
        grids_ids = np.random.choice(range(9), m, replace=False)

        child = np.zeros((9, 9), dtype=int)
        for grid_id in range(9):
            i, j = divmod(grid_id, 3)
            if grid_id in grids_ids:
                child[i * 3:(i + 1) * 3, j * 3:(j + 1) * 3] = parent1[i * 3:(i + 1) * 3, j * 3:(j + 1) * 3]
            else:
                child[i * 3:(i + 1) * 3, j * 3:(j + 1) * 3] = parent2[i * 3:(i + 1) * 3, j * 3:(j + 1) * 3]

        # Ensure fixed positions are maintained
        child[fixed_positions] = initial_sudoku[fixed_positions]

        offspring.append(child.flatten())

    return np.array(offspring)



def custom_mutation(offspring, ga_instance):
    for idx in range(offspring.shape[0]):
        individual = offspring[idx].reshape((9, 9))
        # Choose a random 3x3 subgrid
        grid_id = np.random.randint(9)
        i, j = divmod(grid_id, 3)
        subgrid = individual[i * 3:(i + 1) * 3, j * 3:(j + 1) * 3].flatten()

        # Swap two random positions within the subgrid
        pos1, pos2 = np.random.choice(9, 2, replace=False)
        subgrid[pos1], subgrid[pos2] = subgrid[pos2], subgrid[pos1]

        # Ensure fixed positions are maintained
        individual[i * 3:(i + 1) * 3, j * 3:(j + 1) * 3] = subgrid.reshape((3, 3))
        individual[fixed_positions] = initial_sudoku[fixed_positions]

        offspring[idx] = individual.flatten()
    return offspring


ga_instance = pygad.GA(
    num_generations=10000,
    num_parents_mating=20,
    fitness_func=fitness_func,
    sol_per_pop=population_size,
    num_genes=81,
    initial_population=initial_population,
    parent_selection_type="sss",
    crossover_type=custom_crossover,
    mutation_type=custom_mutation,
    mutation_percent_genes=20,
    on_generation=on_generation
)

# Run the Genetic Algorithm
start_time = time.time()
ga_instance.run()
end_time = time.time()

print("Time taken: {0:.2f} seconds".format(end_time - start_time))
# Retrieve and Display Results
solution, solution_fitness, solution_idx = ga_instance.best_solution()
solution = solution.reshape((9, 9)).astype(int)
