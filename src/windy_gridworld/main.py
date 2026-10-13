import argparse
import sys, math, random
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy as dcopy
from pathlib import Path


class WindyGridworld:
    def __init__(self, stochastic=False):
        self.height = 7
        self.width = 10
        self.stochastic = stochastic
        self.wind = [
            [0, 0],
            [0, 0],
            [0, 0],
            [-1, 0],
            [-1, 0],
            [-1, 0],
            [-2, 0],
            [-2, 0],
            [-1, 0],
            [0, 0],
        ]
        self.grid = [
            [dcopy(self.wind[x]) for x in range(self.width)] for _ in range(self.height)
        ]

    def apply_wind(self, state):
        wind = dcopy(self.grid[state[0]][state[1]])
        if self.stochastic:
            wind[0] += random.choice([-1, 0, 1])
        return wind


class RLAgent:
    def __init__(self, moves, function, learning_rate, discount, epsilon):
        self.moves = moves
        self.num_actions = len(moves)
        self.function = function
        self.lr = learning_rate
        self.discount = discount
        self.epsilon = epsilon
        self.q_table = [
            [[-1e10 for _ in range(self.num_actions)] for _ in range(10)]
            for _ in range(7)
        ]
        self.initialize_q()

    def initialize_q(self):
        for y in range(7):
            for x in range(10):
                for a in range(self.num_actions):
                    new_y = y + self.moves[a][0]
                    new_x = x + self.moves[a][1]
                    if 0 <= new_y < 7 and 0 <= new_x < 10:
                        self.q_table[y][x][a] = 0

    def epsilon_greedy_policy(self, state):
        q_values = self.q_table[state[0]][state[1]]
        valid_actions = [q != -1e10 for q in q_values]
        n_valid = sum(valid_actions)
        comm_prob = self.epsilon / n_valid if n_valid > 0 else 0

        policy = [comm_prob if valid else 0 for valid in valid_actions]
        max_index = np.argmax(q_values)
        policy[max_index] += 1.0 - self.epsilon
        return policy

    def select_action(self, state):
        policy = self.epsilon_greedy_policy(state)
        return random.choices(range(self.num_actions), weights=policy)[0], policy

    def update_q(
        self, state, action, reward, next_state, next_action=None, next_policy=None
    ):
        old_q = self.q_table[state[0]][state[1]][action]
        next_q_values = self.q_table[next_state[0]][next_state[1]]

        if self.function == "Q-Learning":
            target = reward + self.discount * max(next_q_values)
        elif self.function == "Sarsa":
            target = reward + self.discount * next_q_values[next_action]
        elif self.function == "Expected Sarsa":
            target = reward + self.discount * sum(
                p * q for p, q in zip(next_policy, next_q_values)
            )
        else:
            raise ValueError(f"Unknown function: {self.function}")

        self.q_table[state[0]][state[1]][action] = (
            1 - self.lr
        ) * old_q + self.lr * target


class RLRunner:
    def __init__(self, env, agent, start, end, num_episodes):
        self.env = env
        self.agent = agent
        self.start = start
        self.end = end
        self.num_episodes = num_episodes

    def run(self):
        total_rewards = []
        total_timesteps = [0]
        steps = 0

        for _ in range(self.num_episodes):
            state = self.start
            reward_sum = 0

            action, policy = self.agent.select_action(state)

            while state != self.end:
                wind = self.env.apply_wind(state)
                move = self.agent.moves[action]
                next_state = (
                    max(min(state[0] + move[0] + wind[0], 6), 0),
                    max(min(state[1] + move[1] + wind[1], 9), 0),
                )

                reward = -1 if next_state != self.end else 0
                reward_sum += reward

                next_action, next_policy = self.agent.select_action(next_state)

                self.agent.update_q(
                    state,
                    action,
                    reward,
                    next_state,
                    next_action=next_action if self.agent.function == "Sarsa" else None,
                    next_policy=(
                        next_policy if self.agent.function == "Expected Sarsa" else None
                    ),
                )

                state = next_state
                action = next_action
                policy = next_policy
                steps += 1

            total_rewards.append(reward_sum)
            total_timesteps.append(steps)

        return self.agent.q_table, total_rewards, total_timesteps

    def extract_path(self):
        path = [self.start]
        actions = []
        state = self.start
        while state != self.end:
            wind = self.env.grid[state[0]][state[1]]
            action_index = np.argmax(self.agent.q_table[state[0]][state[1]])
            move = self.agent.moves[action_index]
            state = (
                max(min(state[0] + move[0] + wind[0], 6), 0),
                max(min(state[1] + move[1] + wind[1], 9), 0),
            )
            path.append(state)
            actions.append(move)
        return path, actions


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Windy Gridworld RL experiment runner")

    normMoves = [[0, 1], [-1, 0], [0, -1], [1, 0]]
    kingMoves = [[0, 1], [-1, 1], [-1, 0], [-1, -1], [0, -1], [1, -1], [1, 0], [1, 1]]
    options = [
        ("Sarsa", normMoves, False, "Non-stochastic Sarsa with 4 moves"),
        ("Sarsa", kingMoves, False, "Non-stochastic Sarsa with 8 moves"),
        ("Sarsa", kingMoves, True, "Stochastic Sarsa with 8 moves"),
        (
            "Expected Sarsa",
            normMoves,
            False,
            "Non-stochastic Expected Sarsa with 4 moves",
        ),
        ("Q-Learning", normMoves, False, "Non-stochastic Q-Learning with 4 moves"),
    ]
    start, end = (3, 0), (3, 7)

    parser.add_argument(
        "--episodes",
        type=int,
        default=170,
        help="Number of episodes to train (default: 170)",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.1,
        help="Exploration rate (epsilon) (default: 0.1)",
    )
    parser.add_argument(
        "--discount",
        type=float,
        default=1.0,
        help="Discount factor (gamma) (default: 1.0)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.4,
        help="Learning rate (alpha) (default: 0.4)",
    )

    args = parser.parse_args()

    num_episodes = args.episodes
    epsilon = args.epsilon
    discount = args.discount
    learning_rate = args.learning_rate

    for function, moves, stochastic, label in options:
        q_avgs, r_avgs, t_avgs = None, None, None
        for seed in range(10):
            random.seed(seed)
            env = WindyGridworld(stochastic=stochastic)
            agent = RLAgent(moves, function, learning_rate, discount, epsilon)
            runner = RLRunner(env, agent, start, end, num_episodes)

            Q, rewards, steps = runner.run()

            if q_avgs is None:
                q_avgs = np.array(Q) / 10.0
                r_avgs = np.array(rewards) / 10.0
                t_avgs = np.array(steps) / 10.0
            else:
                q_avgs += np.array(Q) / 10.0
                r_avgs += np.array(rewards) / 10.0
                t_avgs += np.array(steps) / 10.0

        if not stochastic:
            print(f"\n{label}")
            runner.agent.q_table = q_avgs  # For path visualization
            path, actions = runner.extract_path()
            print(path)
            print(actions)

        plt.plot(t_avgs, range(len(t_avgs)), label=label)

    plt.legend()
    plt.show()
    output = Path(__file__).resolve().parents[2] / "plots" / "windy_gridworld"
    output.mkdir(parents=True, exist_ok=True)
    plt.savefig(output / "averaged_comparison.png")
