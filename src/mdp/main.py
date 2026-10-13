#!/usr/bin/env python3
# main.py
import argparse
import numpy as np
from pathlib import Path
import tempfile
import time

if __package__:
    from .core.solver import MDPSolver
    from .core.encoder import Encoder
    from .core.decoder import Decoder
    from .core.utils import display_path_on_grid
else:
    from core.solver import MDPSolver
    from core.encoder import Encoder
    from core.decoder import Decoder
    from core.utils import display_path_on_grid


class MDPPipeline:
    """
    Core MDP solving pipeline that can be reused by different scripts.
    """

    def __init__(self, algorithm, temp_dir=None, keep_tmp=False):
        self.algorithm = algorithm
        self.temp_dir = (
            temp_dir
            if temp_dir is not None
            else Path(__file__).resolve().parent / "tmp"
        )
        self.keep_tmp = keep_tmp

        self._workspace = None

    def _create_workspace(self):
        Path(self.temp_dir).mkdir(parents=True, exist_ok=True)
        if self.keep_tmp:
            work_dir = Path(tempfile.mkdtemp(prefix="mdp-", dir=self.temp_dir))
            print(f"Keeping temporary files in: {work_dir}")
        else:
            self._workspace = tempfile.TemporaryDirectory(
                prefix="mdp-", dir=self.temp_dir
            )
            work_dir = Path(self._workspace.name)
        self.mdp_path = str(work_dir / "mdp.txt")
        self.value_path = str(work_dir / "value.txt")
        self.policy_path = str(work_dir / "policy.txt")

    def solve_grid(self, grid_file, visualize=True):
        """
        Solve a maze grid using the MDP approach.

        Args:
            grid_file: Path to the grid file
            visualize: Whether to display the path visually

        Returns:
            tuple: (V, pi, actions, time_taken)
                V: Value function
                pi: Policy
                actions: List of actions forming the path
                time_taken: Time taken to solve
        """
        self._create_workspace()
        start_time = time.time()

        try:
            # Step 1: Encode grid to MDP
            print("Encoding grid to MDP...")
            encoder = Encoder(grid_file)
            mdp_data = encoder.maze
            with open(self.mdp_path, "w") as f:
                f.write(mdp_data)

            # Step 2: Solve MDP
            print(f"Solving MDP using {self.algorithm.upper()}...")
            solver = MDPSolver(mdp_input_path=self.mdp_path, algorithm=self.algorithm)
            V, pi = solver.solve()

            # Write value and policy files
            with open(self.value_path, "w") as f:
                f.writelines(f"{v}\n" for v in V)

            with open(self.policy_path, "w") as f:
                f.writelines(f"{i} {p}\n" for i, p in enumerate(pi))

            # Step 3: Decode policy to path
            print("Decoding policy to path...")
            decoder = Decoder(grid_file, self.value_path, self.policy_path)
            actions = decoder.decode()

            # Format path as a string for display
            path = " ".join(actions)
            print("\nPATH:")
            print(path)

            # Step 4: Visualize path on grid if requested
            if visualize:
                grid = np.loadtxt(grid_file, dtype=int)
                display_path_on_grid(grid, path)

            time_taken = time.time() - start_time
            print(f"Time taken: {time_taken:.3f} sec")

            return V, pi, actions, time_taken

        finally:
            if not self.keep_tmp:
                self.cleanup()

    def cleanup(self):
        """Clean only the private temporary workspace owned by this pipeline."""
        if self._workspace is not None:
            self._workspace.cleanup()
            self._workspace = None


def main(grid_file, algorithm, keep_tmp=False):
    # Create and use the pipeline
    pipeline = MDPPipeline(algorithm, keep_tmp=keep_tmp)
    pipeline.solve_grid(grid_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate, solve, and decode an MDP from a maze grid."
    )
    parser.add_argument("--grid", required=True, help="Path to grid .txt file")
    parser.add_argument(
        "--algorithm",
        choices=["vi", "pi", "lp"],
        default="pi",
        help="Algorithm to use: vi (Value Iteration), pi (Policy Iteration), lp (Linear Programming)",
    )
    parser.add_argument(
        "--keep-tmp", action="store_true", help="Keep temporary files after execution"
    )
    args = parser.parse_args()

    main(args.grid, args.algorithm, args.keep_tmp)
