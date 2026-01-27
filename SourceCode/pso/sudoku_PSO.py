import numpy as np
import pyswarms as ps

class SudokuPSOSolver:
    def __init__(self, initial_sudoku):
        self.initial_grid = initial_sudoku.copy()
        self.blocks_missing = []  # Brakujące cyfry dla każdego bloku 3x3
        self.empty_indices = []   # Pozycje (r, c) pustych pól dla każdego bloku
        
        # Analiza struktury 3x3
        for r_offset in range(0, 9, 3):
            for c_offset in range(0, 9, 3):
                block = self.initial_grid[r_offset:r_offset+3, c_offset:c_offset+3]
                
                # Znajdź brakujące cyfry w bloku
                missing = [n for n in range(1, 10) if n not in block]
                self.blocks_missing.append(missing)
                
                # Znajdź indeksy tych brakujących cyfr
                empties = []
                for i in range(3):
                    for j in range(3):
                        if block[i, j] == 0:
                            empties.append((r_offset + i, c_offset + j))
                self.empty_indices.append(empties)
        
        self.total_dimensions = sum(len(b) for b in self.blocks_missing)

    def reconstruct_grid(self, particle_pos):
        """
        Przekształca wektor ciągły PSO na poprawną planszę Sudoku.
        Wykorzystuje sortowanie (argsort) do stworzenia permutacji.
        """
        grid = self.initial_grid.copy()
        cursor = 0
        
        for block_idx, missing_vals in enumerate(self.blocks_missing):
            n_missing = len(missing_vals)
            if n_missing == 0:
                continue
                
            # Wycinamy fragment pozycji odpowiadający temu blokowi
            block_pos = particle_pos[cursor : cursor + n_missing]
            
            # Klucz do sukcesu: sortujemy brakujące wartości według wag z PSO
            # Dzięki temu zawsze mamy unikalne cyfry w bloku 3x3
            ranking = np.argsort(block_pos)
            for i, (r, c) in enumerate(self.empty_indices[block_idx]):
                grid[r, c] = missing_vals[ranking[i]]
                
            cursor += n_missing
        return grid

    def fitness_func(self, swarm_pos):
        """
        Funkcja celu: minimalizujemy sumę kwadratów brakujących unikalnych cyfr 
        w wierszach i kolumnach (bloki 3x3 są zawsze poprawne).
        """
        n_particles = swarm_pos.shape[0]
        penalties = np.zeros(n_particles)

        for i in range(n_particles):
            grid = self.reconstruct_grid(swarm_pos[i])
            score = 0
            
            # Sprawdzanie wierszy i kolumn
            for idx in range(9):
                # Kara za duplikaty w wierszu
                row_err = 9 - len(np.unique(grid[idx, :]))
                score += row_err**2
                
                # Kara za duplikaty w kolumnie
                col_err = 9 - len(np.unique(grid[:, idx]))
                score += col_err**2
                
            penalties[i] = score
        return penalties

    def solve(self, n_particles=100, iters=1500):
        # Parametry PSO: c1-osobiste, c2-społeczne, w-bezwładność
        # Nieco wyższe c1 pozwala cząsteczkom bardziej eksplorować własne ścieżki
        options = {'c1': 1.4, 'c2': 0.7, 'w': 0.8}

        # Granice: 0-1 (wartości nie mają znaczenia, liczy się ich relacja/ranking)
        bounds = (np.zeros(self.total_dimensions), np.ones(self.total_dimensions))

        optimizer = ps.single.GlobalBestPSO(
            n_particles=n_particles,
            dimensions=self.total_dimensions,
            options=options,
            bounds=bounds
        )

        # Optymalizacja
        best_cost, best_pos = optimizer.optimize(self.fitness_func, iters=iters)
        
        return self.reconstruct_grid(best_pos), best_cost

def run_pso(initial_grid):
    solver = SudokuPSOSolver(initial_grid)
    solution, score = solver.solve(n_particles=200, iters=2000)
    
    print(f"\nFinalny koszt (błędy): {score}")
    return solution
