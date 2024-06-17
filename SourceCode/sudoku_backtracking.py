import numpy as np
import matplotlib.pyplot as plt
import time

def is_valid(board, row, col, num):
    for i in range(9):
        if board[row][i] == num or board[i][col] == num:
            return False

    start_row, start_col = 3 * (row // 3), 3 * (col // 3)
    for i in range(3):
        for j in range(3):
            if board[start_row + i][start_col + j] == num:
                return False

    return True

def solve_sudoku(board):
    empty = find_empty(board)
    if not empty:
        return True

    row, col = empty
    for num in range(1, 10):
        if is_valid(board, row, col, num):
            board[row][col] = num

            if solve_sudoku(board):
                return True

            board[row][col] = 0

    return False

def find_empty(board):
    for i in range(9):
        for j in range(9):
            if board[i][j] == 0:
                return (i, j)
    return None

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

    plt.savefig('sudoku_backtracking_hard.png')
    plt.show()

# easy
initial_sudoku = np.array([
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
fixed_positions = initial_sudoku != 0

# medium
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
# fixed_positions = initial_sudoku != 0

# hard
# initial_sudoku = np.array([
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
# fixed_positions = initial_sudoku != 0

solution = initial_sudoku.copy()


start_time = time.time()
if solve_sudoku(solution):
    end_time = time.time()
    print(f"Sudoku solved in {end_time - start_time:.4f} seconds")
    plot_sudoku_pair(initial_sudoku, solution, correct_solution, fixed_positions)

else:
    print("No solution exists for the given Sudoku puzzle")
