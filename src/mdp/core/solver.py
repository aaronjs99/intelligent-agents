#!/usr/bin/env python3
# solver.py
import os
import numpy as np


class MDPSolver:
    def __init__(
        self, mdp_input_path, algorithm, policy_output_path=None, value_output_path=None
    ):
        self.mdp_input_path = mdp_input_path
        self.algorithm = algorithm
        self.policy_output_path = policy_output_path
        self.value_output_path = value_output_path
        self.tolerance = 1e-12
        self._parse_mdp()

        self.methods = {
            "vi": self._value_iteration,
            "lp": self._linear_programming,
            "pi": self._policy_iteration,
        }

    def _parse_mdp(self):
        with open(self.mdp_input_path) as f:
            lines = [line.strip().split() for line in f]

        self.mdp = {}
        for tokens in lines:
            key = tokens[0]
            if key == "transition":
                if "transition" not in self.mdp:
                    self.mdp["transition"] = [
                        [[] for _ in range(self.mdp["numActions"])]
                        for _ in range(self.mdp["numStates"])
                    ]
                i, j = int(tokens[1]), int(tokens[2])
                self.mdp["transition"][i][j].append(
                    (int(tokens[3]), float(tokens[4]), float(tokens[5]))
                )
            elif key == "mdptype":
                self.mdp["type"] = tokens[1]
            elif key == "end":
                self.mdp["end"] = list(map(int, tokens[1:]))
            elif key == "discount":
                self.mdp["discount"] = float(tokens[1])
            else:
                self.mdp[key] = int(tokens[1])

        self.num_states = self.mdp["numStates"]
        self.num_actions = self.mdp["numActions"]
        self.transition = self.mdp["transition"]
        self.discount = self.mdp["discount"]

    def _compute_q(self, V, s, a):
        return sum(
            p * (r + self.discount * V[s_]) for s_, r, p in self.transition[s][a]
        )

    def _best_action(self, V, s):
        return int(
            np.argmax([self._compute_q(V, s, a) for a in range(self.num_actions)])
        )

    def _evaluate_policy(self, pi):
        V = np.zeros(self.num_states)
        while True:
            new_V = np.array(
                [self._compute_q(V, s, pi[s]) for s in range(self.num_states)]
            )
            if np.linalg.norm(new_V - V) < self.tolerance:
                break
            V = new_V
        return new_V

    def _value_iteration(self):
        V = np.zeros(self.num_states)
        while True:
            new_V = np.array(
                [
                    max(self._compute_q(V, s, a) for a in range(self.num_actions))
                    for s in range(self.num_states)
                ]
            )
            if np.linalg.norm(new_V - V) < self.tolerance:
                break
            V = new_V
        pi = np.array([self._best_action(V, s) for s in range(self.num_states)])
        return V, pi

    def _linear_programming(self):
        import pulp

        V_vars = [pulp.LpVariable(f"V{i}") for i in range(self.num_states)]
        prob = pulp.LpProblem("MDP_LP", pulp.LpMinimize)
        prob += pulp.lpSum(V_vars)

        for s in range(self.num_states):
            for a in range(self.num_actions):
                trans = self.transition[s][a]
                if trans:
                    prob += V_vars[s] >= pulp.lpSum(
                        p * (r + self.discount * V_vars[s_]) for s_, r, p in trans
                    )

        if "end" in self.mdp:
            for terminal_state in self.mdp["end"]:
                prob += V_vars[terminal_state] == 0

        prob.solve(pulp.PULP_CBC_CMD(msg=0))
        if pulp.LpStatus[prob.status] != "Optimal":
            print("⚠️ LP did not solve optimally:", pulp.LpStatus[prob.status])
        V = np.array([round(v.varValue, 6) for v in V_vars])
        pi = np.array([self._best_action(V, s) for s in range(self.num_states)])
        return V, pi

    def _policy_iteration(self):
        pi = np.zeros(self.num_states, dtype=int)
        while True:
            V = self._evaluate_policy(pi)
            new_pi = np.array([self._best_action(V, s) for s in range(self.num_states)])
            if np.array_equal(pi, new_pi):
                break
            pi = new_pi
        return V, pi

    def solve(self):
        return self.methods[self.algorithm]()

    def _write_list(self, path, values, fmt="{:.6f}"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            f.writelines(f"{fmt.format(v)}\n" for v in values)

    def write_output(self, V, pi):
        if self.value_output_path:
            self._write_list(self.value_output_path, V)
        if self.policy_output_path:
            self._write_list(self.policy_output_path, map(int, pi), fmt="{}")
        if not self.value_output_path and not self.policy_output_path:
            for v, p in zip(V, pi):
                print(f"{v:.6f}\t{p}")
