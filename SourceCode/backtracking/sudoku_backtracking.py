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


def run_backtracking(board):
    solution = board.copy()
    if solve_sudoku(solution):
        return solution
    return None
