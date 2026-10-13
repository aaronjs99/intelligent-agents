#!/usr/bin/env python3
# decoder.py
from .grid import Grid, DIRECTIONS, build_maze, manhattan, maze_transitions, neighbors


class Decoder:
    def __init__(self, gridfile, value_path, policy_path):
        self.gridfile = gridfile
        self.grid = Grid(gridfile)
        self.value_path = value_path
        self.policy_path = policy_path
        self.value, self.policy = self.load_value_and_policy()
        self.states, self.start, self.end = self.build_maze()

    def load_value_and_policy(self):
        value, policy = [], []
        with open(self.value_path) as f:
            out = f.readlines()
            out = [out[i].split() for i in range(len(out))]
            for i in range(len(out)):
                value.append(float(out[i][0]))
        with open(self.policy_path) as f:
            out = f.readlines()
            out = [out[i].split() for i in range(len(out))]
            for i in range(len(out)):
                policy.append(int(out[i][1]))
        return value, policy

    manhattan = staticmethod(manhattan)

    def build_maze(self):
        states, start, ends = build_maze(self.grid.grid)
        # Preserve the decoder's historical last-goal semantics.
        return states, start, ends[-1] if ends else None

    def get_maze_data(self):
        return (
            len(self.states),
            4,
            self.start,
            self.end,
            maze_transitions(self.states, self.end),
            "episodic",
            0.9,
        )

    def decode(self):
        translated = list(DIRECTIONS)
        state_num = self.start
        state = self.states[state_num]
        actions = []

        state_to_idx = {state: idx for idx, state in enumerate(self.states)}

        while state_num != self.end:
            action_num = self.policy[state_num]
            action = translated[action_num]
            actions.append(action)

            state = neighbors(state)[action_num]
            try:
                state_num = state_to_idx[state]
            except KeyError:
                raise ValueError(f"Invalid state transition to {state}")

        return actions
