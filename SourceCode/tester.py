import numpy as np
import time
import sys
import os
import argparse
import matplotlib.pyplot as plt

# Add subdirectories to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, "aco"))
sys.path.append(os.path.join(current_dir, "backtracking"))
sys.path.append(os.path.join(current_dir, "ga"))
sys.path.append(os.path.join(current_dir, "pso"))

try:
    from sudoku_ACO import run_aco
    from sudoku_backtracking import run_backtracking
    from sudoku_GA import run_ga
    from sudoku_PSO import run_pso
except ImportError as e:
    print(f"Error importing solvers: {e}")
    sys.exit(1)

# --- Define Puzzles ---

# Easy
easy_problem = np.array(
    [
        [9, 0, 2, 0, 7, 8, 4, 0, 0],
        [1, 8, 5, 0, 0, 0, 7, 6, 0],
        [0, 7, 0, 5, 0, 0, 0, 0, 0],
        [7, 5, 0, 0, 0, 6, 0, 8, 4],
        [4, 0, 6, 0, 0, 0, 1, 5, 7],
        [0, 1, 0, 0, 4, 5, 3, 9, 6],
        [0, 2, 0, 0, 0, 0, 5, 0, 9],
        [6, 0, 9, 0, 0, 7, 8, 3, 0],
        [0, 0, 0, 9, 8, 4, 0, 7, 0],
    ],
    dtype=int,
)

easy_solution = np.array(
    [
        [9, 6, 2, 3, 7, 8, 4, 1, 5],
        [1, 8, 5, 4, 2, 9, 7, 6, 3],
        [3, 7, 4, 5, 6, 1, 9, 2, 8],
        [7, 5, 3, 1, 9, 6, 2, 8, 4],
        [4, 9, 6, 8, 3, 2, 1, 5, 7],
        [2, 1, 8, 7, 4, 5, 3, 9, 6],
        [8, 2, 7, 6, 1, 3, 5, 4, 9],
        [6, 4, 9, 2, 5, 7, 8, 3, 1],
        [5, 3, 1, 9, 8, 4, 6, 7, 2],
    ],
    dtype=int,
)

# Medium
medium_problem = np.array(
    [
        [9, 6, 0, 0, 0, 0, 4, 0, 0],
        [1, 8, 5, 4, 2, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 9, 0, 8],
        [5, 3, 0, 9, 8, 0, 6, 0, 0],
        [0, 4, 9, 0, 0, 0, 0, 0, 1],
        [8, 2, 7, 0, 0, 0, 0, 4, 9],
        [7, 5, 0, 1, 0, 6, 0, 8, 0],
        [0, 0, 6, 8, 3, 0, 0, 5, 0],
        [0, 1, 8, 7, 0, 0, 3, 0, 6],
    ],
    dtype=int,
)

medium_solution = np.array(
    [
        [9, 6, 2, 3, 7, 8, 4, 1, 5],
        [1, 8, 5, 4, 2, 9, 7, 6, 3],
        [3, 7, 4, 5, 6, 1, 9, 2, 8],
        [5, 3, 1, 9, 8, 4, 6, 7, 2],
        [6, 4, 9, 2, 5, 7, 8, 3, 1],
        [8, 2, 7, 6, 1, 3, 5, 4, 9],
        [7, 5, 3, 1, 9, 6, 2, 8, 4],
        [4, 9, 6, 8, 3, 2, 1, 5, 7],
        [2, 1, 8, 7, 4, 5, 3, 9, 6],
    ],
    dtype=int,
)

# # Hard
# hard_problem = np.array(
#     [
#         [5, 0, 6, 1, 0, 2, 0, 0, 0],
#         [0, 0, 0, 6, 5, 0, 0, 1, 7],
#         [8, 1, 0, 0, 0, 0, 0, 5, 0],
#         [0, 0, 0, 2, 0, 0, 0, 9, 0],
#         [9, 0, 0, 5, 0, 7, 0, 0, 8],
#         [0, 5, 1, 0, 3, 9, 0, 4, 0],
#         [0, 0, 8, 0, 0, 0, 4, 0, 9],
#         [7, 6, 5, 9, 8, 4, 0, 0, 0],
#         [0, 0, 9, 0, 0, 0, 0, 0, 0],
#     ],
#     dtype=int,
# )

# hard_solution = np.array(
#     [
#         [5, 7, 6, 1, 9, 2, 3, 8, 4],
#         [4, 9, 3, 6, 5, 8, 2, 1, 7],
#         [8, 1, 2, 4, 7, 3, 9, 5, 6],
#         [3, 8, 7, 2, 4, 6, 5, 9, 1],
#         [9, 2, 4, 5, 1, 7, 6, 3, 8],
#         [6, 5, 1, 8, 3, 9, 7, 4, 2],
#         [1, 3, 8, 7, 2, 5, 4, 6, 9],
#         [7, 6, 5, 9, 8, 4, 1, 2, 3],
#         [2, 4, 9, 3, 6, 1, 8, 7, 5],
#     ],
#     dtype=int,
# )
hard_problem = np.array(
    [
        [0, 0, 0, 0, 0, 1, 2, 3, 0],
        [1, 2, 3, 0, 0, 8, 0, 4, 0],
        [8, 0, 4, 0, 0, 7, 6, 5, 0],
        [7, 6, 5, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 2, 3],
        [0, 1, 2, 3, 0, 0, 8, 0, 4],
        [0, 8, 0, 4, 0, 0, 7, 6, 5],
        [0, 7, 6, 5, 0, 0, 0, 0, 0],
    ],
    dtype=int,
)

hard_solution = np.array(
    [
        [6, 5, 7, 9, 4, 1, 2, 3, 8],
        [1, 2, 3, 6, 5, 8, 9, 4, 7],
        [8, 9, 4, 2, 3, 7, 6, 5, 1],
        [7, 6, 5, 1, 2, 3, 4, 8, 9],
        [2, 3, 1, 8, 9, 4, 5, 7, 6],
        [9, 4, 8, 7, 6, 5, 1, 2, 3],
        [5, 1, 2, 3, 7, 6, 8, 9, 4],
        [3, 8, 9, 4, 1, 2, 7, 6, 5],
        [4, 7, 6, 5, 8, 9, 3, 1, 2],
    ],
    dtype=int,
)
TEST_CASES = {
    "easy": (easy_problem, easy_solution),
    "medium": (medium_problem, medium_solution),
    "hard": (hard_problem, hard_solution),
}

SOLVERS = {
    "aco": run_aco,
    "backtracking": run_backtracking,
    "ga": run_ga,
    "pso": run_pso,
}


def plot_sudoku_result(
    initial_sudoku, solution, correct_solution, solver_name, difficulty
):
    fixed_positions = initial_sudoku != 0

    # Save in appropriate subfolder
    filename = os.path.join(
        current_dir, solver_name, f"sudoku_{solver_name}_{difficulty}.png"
    )

    fig, axs = plt.subplots(1, 2, figsize=(20, 10))
    titles = ["Initial Sudoku Puzzle", "Computed Solution"]
    sudokus = [initial_sudoku, solution]

    for ax, sudoku, title in zip(axs, sudokus, titles):
        ax.set_title(title, fontsize=20, fontweight="bold")
        ax.axis("off")

        # Draw grid lines
        for i in range(10):
            lw = 2 if i % 3 == 0 else 1
            ax.plot([0, 9], [i, i], color="black", linewidth=lw)
            ax.plot([i, i], [0, 9], color="black", linewidth=lw)

        # Fill numbers
        for i in range(9):
            for j in range(9):
                if sudoku[i, j] != 0:
                    val = sudoku[i, j]
                    if fixed_positions[i, j]:
                        ax.text(
                            j + 0.5,
                            i + 0.5,
                            str(val),
                            va="center",
                            ha="center",
                            color="black",
                            fontsize=16,
                            fontweight="bold",
                        )
                    else:
                        color = "blue"
                        if (
                            title == "Computed Solution"
                            and correct_solution is not None
                        ):
                            if val == correct_solution[i, j]:
                                color = "green"
                            else:
                                color = "red"
                        ax.text(
                            j + 0.5,
                            i + 0.5,
                            str(val),
                            va="center",
                            ha="center",
                            color=color,
                            fontsize=16,
                        )

        ax.invert_yaxis()

    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    print(f"Plot saved to {filename}")


def calculate_accuracy(result, correct_solution):
    if result is None:
        return 0.0

    total = result.size
    correct = np.sum(result == correct_solution)
    return (correct / total) * 100


def run_tests(solver_name, difficulty, iterations):
    if solver_name not in SOLVERS:
        print(f"Unknown solver: {solver_name}. Available: {list(SOLVERS.keys())}")
        return
    if difficulty not in TEST_CASES:
        print(f"Unknown difficulty: {difficulty}. Available: {list(TEST_CASES.keys())}")
        return

    solver_func = SOLVERS[solver_name]
    problem, correct_solution = TEST_CASES[difficulty]

    times = []
    accuracies = []
    last_result_board = None

    print(
        f"Running {solver_name.upper()} on {difficulty.upper()} Sudoku {iterations} times..."
    )

    for i in range(iterations):
        print(f"Run {i+1}/{iterations}...", end="", flush=True)
        start_time = time.time()

        result_board = None
        try:
            if solver_name == "aco":
                result_board = solver_func(problem, timeout=60)
            elif solver_name == "backtracking":
                result_board = solver_func(problem.copy())
            elif solver_name == "ga":
                result_board = solver_func(
                    problem, correct_solution, num_generations=1000
                )

            elif solver_name == "pso":
                result_board = solver_func(problem)
        except Exception as e:

            print(f" Error: {e}")

        if result_board is not None:
            last_result_board = result_board

        end_time = time.time()
        duration = end_time - start_time
        times.append(duration)

        acc = 0.0
        if result_board is not None:
            acc = calculate_accuracy(result_board, correct_solution)
        accuracies.append(acc)

        print(f" Time: {duration:.2f}s, Accuracy: {acc:.1f}%")

    avg_time = np.mean(times)
    avg_acc = np.mean(accuracies)

    print("\n" + "=" * 40)
    print(f"RESULTS: {solver_name.upper()} - {difficulty.upper()}")
    print("=" * 40)
    print(f"Iterations: {iterations}")
    print(f"Average Time: {avg_time:.4f} seconds")
    print(f"Average Accuracy: {avg_acc:.2f}%")
    print("=" * 40)

    if last_result_board is not None:
        plot_sudoku_result(
            problem, last_result_board, correct_solution, solver_name, difficulty
        )


def main():
    parser = argparse.ArgumentParser(description="Sudoku Solver Tester")
    parser.add_argument(
        "--solver",
        type=str,
        choices=SOLVERS.keys(),
        help="Solver to use: aco, backtracking, ga",
    )
    parser.add_argument(
        "--difficulty",
        type=str,
        choices=TEST_CASES.keys(),
        help="Difficulty: easy, medium, hard",
    )
    parser.add_argument(
        "--iterations", type=int, default=1, help="Number of iterations"
    )

    args = parser.parse_args()

    # Interactive mode if arguments are missing
    if not args.solver or not args.difficulty:
        print("Interactive Mode")
        print("Available solvers:", ", ".join(SOLVERS.keys()))
        s = input("Select solver: ").strip().lower()

        print("Available difficulties:", ", ".join(TEST_CASES.keys()))
        d = input("Select difficulty: ").strip().lower()

        iter_input = input("Number of iterations (default 1): ").strip()
        n = int(iter_input) if iter_input else 1

        run_tests(s, d, n)
    else:
        run_tests(args.solver, args.difficulty, args.iterations)


if __name__ == "__main__":
    main()
