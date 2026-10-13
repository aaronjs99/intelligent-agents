#!/usr/bin/env python3
# grid.py


class Grid:
    def __init__(self, gridfile):
        self.gridfile = gridfile
        self.grid = self.load_grid()

    def load_grid(self):
        try:
            with open(self.gridfile) as f:
                grid = f.readlines()
            return [[int(j) for j in grid[i].strip().split()] for i in range(len(grid))]
        except FileNotFoundError:
            print(f"Error: {self.gridfile} not found.")
            raise
        except ValueError as e:
            print(f"Error reading the grid file {self.gridfile}: {e}")
            raise


# Policy action order is part of the course file format.
DIRECTIONS = {"N": (-1, 0), "S": (1, 0), "E": (0, 1), "W": (0, -1)}


def manhattan(point1, point2):
    return abs(point1[0] - point2[0]) + abs(point1[1] - point2[1])


def neighbors(state):
    return [(state[0] + dy, state[1] + dx) for dy, dx in DIRECTIONS.values()]


def build_maze(grid):
    """Enumerate traversable cells in file order and retain all goal indices."""
    states, ends = [], []
    start = None
    for i, row in enumerate(grid):
        for j, cell in enumerate(row):
            if cell != 1:
                states.append((i, j))
            if cell == 2:
                start = len(states) - 1
            if cell == 3:
                ends.append(len(states) - 1)
    return states, start, ends


def maze_transitions(states, end):
    """Build deterministic transitions without changing reward/terminal semantics."""
    indices = {state: i for i, state in enumerate(states)}
    end_state = states[end]
    transitions = []
    for i, state in enumerate(states):
        if i == end:
            continue
        for action, neighbor in enumerate(neighbors(state)):
            if neighbor == end_state:
                transitions.append((i, action, end, 1000000, 1.0))
            elif neighbor in indices:
                transitions.append((i, action, indices[neighbor], -1, 1.0))
            else:
                transitions.append((i, action, i, -1000000, 1.0))
    return transitions
