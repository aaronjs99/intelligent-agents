"""Adaptive decision experiments and preserved intelligent-agents examples."""

import argparse
import logging
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

ROOT = Path(__file__).resolve().parent
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def project_path(path: str) -> str:
    """Treat relative input and output paths as relative to this repository."""
    candidate = Path(path)
    return str(candidate if candidate.is_absolute() else ROOT / candidate)


@dataclass
class BanditConfig:
    instance: str
    algorithm: str
    rseed: int = 0
    epsilon: Optional[float] = None
    horizon: int = 1000


@dataclass
class WindyGridworldConfig:
    episodes: int = 170
    epsilon: float = 0.1
    discount: float = 1.0
    learning_rate: float = 0.4


@dataclass
class MDPGenerateConfig:
    output_file: str
    num_states: int = 10
    num_actions: int = 5
    gamma: float = 0.95
    mdptype: str = "episodic"
    rseed: int = 42


@dataclass
class MDPSolveConfig:
    grid: str
    algorithm: str = "pi"
    keep_tmp: bool = False


@dataclass
class MDPVisualizeConfig:
    grid_file: str
    output_file: str
    path_file: Optional[str] = None


class SubcommandRunner:
    def __init__(self, verbose: bool):
        self.verbose = verbose

    def _log(self, level: int, message: str) -> None:
        if self.verbose:
            logger.log(level, message)

    def _execute(self, command: Sequence[str]) -> None:
        self._log(logging.DEBUG, f"Running: {' '.join(command)}")
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        self._log(
            logging.DEBUG, f"Command finished with exit code: {result.returncode}"
        )
        if self.verbose and result.stdout:
            print(result.stdout)
        if self.verbose and result.stderr:
            print(result.stderr)


class BanditRunner(SubcommandRunner):
    def run(self, config: BanditConfig) -> None:
        command = [
            sys.executable,
            str(ROOT / "src" / "bandits" / "main.py"),
            f"--instance={project_path(config.instance)}",
            f"--algorithm={config.algorithm}",
            f"--rseed={config.rseed}",
            f"--horizon={config.horizon}",
        ]
        if config.epsilon is not None:
            command.append(f"--epsilon={config.epsilon}")
        self._execute(command)


class WindyGridworldRunner(SubcommandRunner):
    def run(self, config: WindyGridworldConfig) -> None:
        self._execute(
            [
                sys.executable,
                str(ROOT / "src" / "windy_gridworld" / "main.py"),
                f"--episodes={config.episodes}",
                f"--epsilon={config.epsilon}",
                f"--discount={config.discount}",
                f"--learning-rate={config.learning_rate}",
            ]
        )


class MDPGenerateRunner(SubcommandRunner):
    def run(self, config: MDPGenerateConfig) -> None:
        self._execute(
            [
                sys.executable,
                str(ROOT / "src" / "mdp" / "generate.py"),
                f"--num_states={config.num_states}",
                f"--num_actions={config.num_actions}",
                f"--gamma={config.gamma}",
                f"--mdptype={config.mdptype}",
                f"--rseed={config.rseed}",
                f"--output_file={project_path(config.output_file)}",
            ]
        )


class MDPSolveRunner(SubcommandRunner):
    def run(self, config: MDPSolveConfig) -> None:
        command = [
            sys.executable,
            str(ROOT / "src" / "mdp" / "main.py"),
            f"--grid={project_path(config.grid)}",
            f"--algorithm={config.algorithm}",
        ]
        if config.keep_tmp:
            command.append("--keep-tmp")
        self._execute(command)


class MDPVisualizeRunner(SubcommandRunner):
    def run(self, config: MDPVisualizeConfig) -> None:
        command = [
            sys.executable,
            str(ROOT / "src" / "mdp" / "visualize.py"),
            f"--grid_file={project_path(config.grid_file)}",
            f"--output_file={project_path(config.output_file)}",
        ]
        if config.path_file:
            command.append(f"--path_file={project_path(config.path_file)}")
        self._execute(command)


def config_from_args(args: argparse.Namespace):
    config_types = {
        "bandits": BanditConfig,
        "windy": WindyGridworldConfig,
        "generate_mdp": MDPGenerateConfig,
        "solve_mdp": MDPSolveConfig,
        "visualize_mdp": MDPVisualizeConfig,
    }
    config_type = config_types[args.command]
    values = {
        name: value
        for name, value in vars(args).items()
        if name in config_type.__annotations__
    }
    return config_type(**values)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Adaptive decision experiments and intelligent-agents examples"
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Show child-program output"
    )
    subparsers = parser.add_subparsers(dest="command")

    from src.bandits.adaptive import add_arguments

    adaptive_parser = subparsers.add_parser(
        "adapt", help="Compare service routing under drift and uncertain advice"
    )
    add_arguments(adaptive_parser)

    bandit_parser = subparsers.add_parser(
        "bandits", help="Run a multi-armed bandit example"
    )
    bandit_parser.add_argument("--instance", default="data/bandits/instances/i-1.txt")
    bandit_parser.add_argument(
        "--algorithm",
        choices=[
            "ucb",
            "kl-ucb",
            "epsilon-greedy",
            "thompson-sampling",
            "thompson-sampling-with-hint",
        ],
        default="ucb",
    )
    bandit_parser.add_argument("--rseed", type=int, default=0)
    bandit_parser.add_argument("--epsilon", type=float, default=0.1)
    bandit_parser.add_argument("--horizon", type=int, default=1000)
    bandit_parser.set_defaults(runner=BanditRunner)

    gridworld_parser = subparsers.add_parser(
        "windy", help="Run a Windy Gridworld example"
    )
    gridworld_parser.add_argument("--episodes", type=int, default=170)
    gridworld_parser.add_argument("--epsilon", type=float, default=0.1)
    gridworld_parser.add_argument("--discount", type=float, default=1.0)
    gridworld_parser.add_argument("--learning-rate", type=float, default=0.4)
    gridworld_parser.set_defaults(runner=WindyGridworldRunner)

    generate_parser = subparsers.add_parser(
        "generate_mdp", help="Generate a random MDP example"
    )
    generate_parser.add_argument("--num_states", type=int, default=10)
    generate_parser.add_argument("--num_actions", type=int, default=5)
    generate_parser.add_argument("--gamma", type=float, default=0.95)
    generate_parser.add_argument(
        "--mdptype", choices=["continuing", "episodic"], default="episodic"
    )
    generate_parser.add_argument("--rseed", type=int, default=42)
    generate_parser.add_argument("--output_file", required=True)
    generate_parser.set_defaults(runner=MDPGenerateRunner)

    solve_parser = subparsers.add_parser(
        "solve_mdp", help="Solve a maze MDP with vi, lp, or pi"
    )
    solve_parser.add_argument("--grid", required=True)
    solve_parser.add_argument("--algorithm", choices=["vi", "pi", "lp"], default="pi")
    solve_parser.add_argument("--keep-tmp", action="store_true")
    solve_parser.set_defaults(runner=MDPSolveRunner)

    visualize_parser = subparsers.add_parser(
        "visualize_mdp", help="Render a maze grid or solution path"
    )
    visualize_parser.add_argument(
        "--grid_file", required=True, help="Path to a grid file"
    )
    visualize_parser.add_argument("--path_file", help="Optional path file")
    visualize_parser.add_argument(
        "--output_file", required=True, help="Output image path"
    )
    visualize_parser.set_defaults(runner=MDPVisualizeRunner)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "adapt":
        from src.bandits.adaptive import run_from_args

        args.output_dir = Path(project_path(str(args.output_dir)))
        if args.scenario is not None:
            args.scenario = Path(project_path(str(args.scenario)))
        try:
            run_from_args(args)
        except (ValueError, KeyError, TypeError, OSError) as error:
            parser.error(str(error))
        except KeyboardInterrupt:
            parser.exit(130, "Interrupted. Completed runs, if any, are preserved.\n")
        return
    if not hasattr(args, "runner"):
        parser.print_help()
        return
    args.runner(args.verbose).run(config_from_args(args))


if __name__ == "__main__":
    main()
