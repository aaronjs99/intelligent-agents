#! /usr/bin/python
# utils.py
import numpy as np
import matplotlib.pyplot as plt

from .grid import DIRECTIONS


def print_maze_with_padding(maze):
    """
    Prints the maze with a border of 1s (walls) on all sides.
    """
    if not maze or not maze[0]:
        raise ValueError("Maze must be a non-empty 2D list")

    rows = len(maze)
    cols = len(maze[0])

    # Print top wall
    print("1 " * (cols + 2))

    for row in maze:
        print("1", end=" ")  # Left wall
        print(" ".join(str(cell) for cell in row), end=" ")
        print("1")  # Right wall

    # Print bottom wall
    print("1 " * (cols + 2))


def display_path_on_grid(grid, path):
    grid = grid.copy()
    path_steps = path.strip().split()
    x, y = np.where(grid == 2)  # Find the start point (2)
    x, y = int(x[0]), int(y[0])

    for step in path_steps:
        dx, dy = DIRECTIONS.get(step, (0, 0))
        x += dx
        y += dy
        if grid[x][y] != 3:  # Don't overwrite goal (3)
            grid[x][y] = 7  # Mark the path with 7

    plt.figure(figsize=(8, 8))
    plt.imshow(grid, cmap="tab10")
    plt.title("Decoded Path on Maze")
    plt.axis("off")
    plt.show()
