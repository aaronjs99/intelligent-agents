"""Encode a maze grid using the historical course MDP format."""

from .grid import Grid, build_maze, manhattan, maze_transitions


class Encoder:
    def __init__(self, gridfile):
        self.gridfile = gridfile
        self.grid = Grid(gridfile)
        self.maze = self.encode(self.grid)

    manhattan = staticmethod(manhattan)

    def encode(self, grid=None):
        if grid is not None:
            self.grid = grid
        states, start, ends = build_maze(self.grid.grid)
        # Preserve the encoder's nearest-goal selection and file-order tie break.
        end = min(ends, key=lambda goal: manhattan(states[goal], states[start]))
        transitions = maze_transitions(states, end)
        lines = [
            f"numStates {len(states)}",
            "numActions 4",
            f"start {start}",
            f"end {end}",
        ]
        lines.extend("transition " + " ".join(map(str, t)) for t in transitions)
        lines.extend(["mdptype episodic", "discount 0.9"])
        return "\n".join(lines) + "\n"
