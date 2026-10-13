#!/usr/bin/env python3
# visualize.py
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

if __package__:
    from .core.grid import DIRECTIONS
else:
    from core.grid import DIRECTIONS


class MazeVisualizer:
    def __init__(self, grid_path, path_path=None, output="gridFile.png"):
        self.grid = np.loadtxt(grid_path, dtype=int)
        self.path_path = path_path
        self.output = output
        self.rows, self.cols = self.grid.shape
        self.start_pos = self._find_start()

    def _find_start(self):
        for i in range(self.rows):
            for j in range(self.cols):
                if self.grid[i][j] == 2:
                    return (i, j)
        return None

    def _apply_path(self):
        x, y = self.start_pos
        self.grid[x][y] = 3

        with open(self.path_path) as f:
            steps = f.readlines()[-1].split()

        for move in steps:
            dy, dx = DIRECTIONS.get(move, (0, 0))
            x += dy
            y += dx
            self.grid[x][y] = 2

        self.grid[x][y] = 4

    def _convert_solution_markers(self):
        for i in range(self.rows):
            for j in range(self.cols):
                if self.grid[i][j] == 2:
                    self.grid[i][j] = 3
                elif self.grid[i][j] == 3:
                    self.grid[i][j] = 4

    def render(self):
        if self.path_path:
            self._apply_path()
        else:
            self._convert_solution_markers()

        plt.figure(figsize=(10, 5))
        plt.imshow(self.grid, cmap=plt.cm.CMRmap, interpolation="nearest")
        plt.xticks([]), plt.yticks([])
        Path(self.output).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(self.output)
        plt.close()
        print(f"Maze image saved to: {self.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize MDP solutions.")
    parser.add_argument(
        "--grid_file",
        type=str,
        default="data/mdp/grids/grid10.txt",
        help="Path to the grid file",
    )
    parser.add_argument(
        "--path_file", type=str, default=None, help="Path to the path file"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="plots/mdp/grid10_unsolved.png",
        help="Path to the output file",
    )
    args = parser.parse_args()

    if args.path_file:
        print(f"Applying path from: {args.path_file}" f" to grid: {args.grid_file}")
        if "unsolved" in args.output_file:
            output_file = args.output_file.replace("unsolved", "solved")
            print(f"Saving solved grid to: {output_file}")
    else:
        print(f"Converting solution markers in grid: {args.grid_file}")
        if "solved" in args.output_file and "unsolved" not in args.output_file:
            output_file = args.output_file.replace("solved", "unsolved")
            print(f"Saving unsolved grid to: {output_file}")

    visualizer = MazeVisualizer(
        grid_path=args.grid_file, path_path=args.path_file, output=args.output_file
    )
    visualizer.render()
