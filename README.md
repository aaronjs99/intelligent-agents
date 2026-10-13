# Intelligent Agents

A small adaptive decision laboratory: **learn which action works, adapt when the
world changes, and check advice against experience.** Its first complete use case
is service routing under changing reliability and stale recommendations.

The original CS747 bandit, maze-planning, and Windy Gridworld implementations remain
available as foundations and coursework references. The routing extension is a
new application and composition of existing techniques with a heuristic advice
gate; it does not claim a novel learning algorithm.

## Try the useful demonstration

```bash
python run.py adapt
```

Three simulated service routes start with success probabilities `[0.85, 0.55,
0.35]`. Halfway through, they become `[0.30, 0.80, 0.55]`. The old recommended
route is now unreliable. Four agents must choose routes and learn only from the
success or failure of their chosen route:

| Agent | What it does | What the comparison reveals |
| --- | --- | --- |
| `stationary-ts` | Thompson sampling with all historical observations | Old evidence can slow adaptation |
| `forgetting-ts` | Discounts old evidence each round | A baseline for adapting to drift |
| `blind-hints` | Combines posterior samples with unchecked advice | Confident stale advice can prevent exploration |
| `checked-hints` | Attenuates the whole advice source when observed outcomes disagree | Whether evidence can limit harm from bad advice |

The command prints mean successful requests, expected regret, its standard error,
and the optimal-route fraction during the last phase. It also prints the regret
saved by checked hints against forgetting in **each** phase. Every invocation
creates a new `results/adaptive-routing/run-<UTC timestamp>-<unique ID>/` folder:

- `report.html`: an offline report with phase and seed filters, scenario details,
  the comparison chart, and links to the data files. Open the printed path in a browser.
- `summary.json`: the exact scenario, seeds, runtime versions, total and per-phase
  metrics, per-seed results, paired comparisons, and limitations (report schema 1).
- `decisions.csv`: every action, reward, expected regret, and hint weight.
- `comparison.png`: cumulative regret and rolling optimal-route choice.

Existing runs, including older reports directly inside the output directory,
remain intact. A bundle appears at its final path only after all four files have
been generated. An export failure cleans up its private staging directory; an
abrupt process termination can leave a hidden staging folder, which is not a
completed run. `--output-dir` selects the **parent** directory for new run folders.

In the HTML report, **regret saved** is forgetting's regret minus the compared
agent's regret on the same seed. Positive values favor the compared agent.
Its standard error is calculated from those paired differences. Individual-seed
views show no estimated standard error. The embedded chart always uses all phases
and seeds; the filters update the table and interpretation below it. The HTML
itself works offline; keep the CSV and PNG beside it for their download links.

Outputs are generated and ignored by Git. The default run uses 1,000 rounds,
seeds 0–9, and a forgetting discount of 0.98. Agents share the same simulated
potential outcomes per seed and paired sampling streams. Lower expected regret
means choosing routes closer to the **current** best route; it is not the noisy
difference between optimal expected reward and observed successes.

Change the advice condition without changing the reward scenario:

```bash
python run.py adapt --advice fresh --output-dir results/fresh
python run.py adapt --advice misleading --output-dir results/misleading
python run.py adapt --advice none --output-dir results/no-advice
```

In one reproducible run of these defaults, mean expected regret was:

| Advice | Stationary TS | Forgetting TS | Blind hints | Checked hints |
| --- | ---: | ---: | ---: | ---: |
| Stale after the change | 143.94 | 58.10 | 250.00 | 40.33 |
| Fresh and exactly accurate | 143.94 | 58.10 | 0.00 | 13.85 |
| Misleading throughout | 143.94 | 58.10 | 375.00 | 63.18 |
| Absent | 143.94 | 58.10 | 58.10 | 58.10 |

These are synthetic measurements, not deployment evidence. Checking has a cost:
it loses to perfect advice and does not beat forgetting alone in the misleading
condition. In the stale run, its total benefit comes from useful advice before
the change: during the degraded phase, checked hints incurred 40.33 regret versus
33.73 for forgetting alone. This run does not show faster adaptation from checking.
NumPy version changes may alter sampling details; the report records Python and
NumPy versions. The JSON and CSV make each run inspectable rather than guaranteeing
these figures across environments.

## Compose an agent, not a framework

| Boundary | Implementation | Contract |
| --- | --- | --- |
| Environment | `src/bandits/environment.py` | `advice(step)`, `step(action, step)` |
| Reward model | `src/bandits/model.py` | estimates, uncertainty, sampling, and feedback update |
| One-step planner | `src/bandits/agents.py` | `choose(model, rng, hint)` |
| Agent | `src/bandits/agents.py` | `act(hint)`, `observe(action, reward)` |
| Experiment | `src/bandits/experiment.py` | `run_episode(environment, agent, name, seed)` and paired comparisons |
| Scenario | `src/bandits/scenario.py`, `validation.py` | strict JSON loading and shared phase validation without simulation |
| Report | `src/bandits/reporting.py`, `report_template.html` | phase/seed diagnostics, paired differences, complete report bundles |
| CLI | `src/bandits/adaptive.py` | scenario loading and a runnable report |

`DecisionEnvironment`, `RewardModel`, `Planner`, and `Agent` are small structural
interfaces. No inheritance registry is required. The agent does not receive true
success probabilities, best-action labels, or evaluation regret. Environment
outcomes carry those metrics to the experiment, which forwards only the observed
reward to the agent.

For an application that already has outcomes, use just the agent. With `src` on
your Python path:

```python
import numpy as np
from bandits import BetaRewardModel, Hint, RewardAgent, ThompsonPlanner

agent = RewardAgent(
    BetaRewardModel(num_actions=3, discount=0.98),
    ThompsonPlanner(advice="checked"),
    np.random.default_rng(42),
)
decision = agent.act(Hint((0.85, 0.55, 0.35), confidence=0.9))
# Dispatch one task through decision.action, then provide its actual result:
agent.observe(decision.action, reward=1)
```

The checked planner treats the hint vector as advice from one source. Once an arm
has five effective observations, disagreement exceeding two model standard
deviations attenuates that source's entire vector. The attenuation scale is 0.05.
These are explicit heuristic defaults, not calibrated credibility guarantees.
The forgetting model discounts successes/failures toward its Beta(1, 1) prior;
under drift those fractional counts are a useful estimate, not an exact Bayesian
posterior for the changing process.

## Bring a scenario

`--scenario path/to/scenario.json` accepts any fixed number of routes and one or
more named phases; their round counts replace `--horizon`, and their hints replace
`--advice`. Check the included normal/outage/recovery scenario without running or
writing any results, then run it:

```bash
python run.py adapt --scenario data/bandits/scenarios/outage.json --validate-only
python run.py adapt --scenario data/bandits/scenarios/outage.json --seeds 5
```

`--validate-only` checks both the scenario and the experiment workload. A custom
scenario looks like:

```json
{
  "phases": [
    {"name": "normal", "rounds": 300, "probabilities": [0.9, 0.6],
     "hint": {"probabilities": [0.9, 0.6], "confidence": 0.8}},
    {"name": "outage", "rounds": 300, "probabilities": [0.2, 0.8],
     "hint": {"probabilities": [0.9, 0.6], "confidence": 0.8}}
  ]
}
```

Omit `hint` (or use `null`) for no advice. Probabilities and hint confidence must be
finite numbers between 0 and 1, phase names must be nonempty and unique, and rounds
must be positive integers. At least two routes are required, with the same route
count in every phase and hint. Misspelled or duplicate fields, strings pretending
to be numbers, booleans, and malformed JSON are rejected with a file and field
path (or JSON line/column). Use UTF-8 JSON. Unsupported characters in a legacy
Windows console are escaped while the UTF-8 report retains the original names.
Relative
scenario and output paths passed through `run.py` are repository-relative. The
direct `python src/bandits/adaptive.py` entrypoint uses caller-relative paths.

This is an in-memory experiment tool. To reject accidental oversized jobs before
allocation, a run is limited to 2,000,000 decisions across all four agents and
seeds, 5,000,000 simulated potential outcomes per seed, 20,000,000 route
evaluations across decisions, and 10,000 agent/phase/seed summary groups. The last
count includes aggregate phases and seeds. Errors explain which limit was exceeded
and whether to reduce rounds, routes, phases, or seed replications. These limits
bound work; they do not promise a particular runtime or peak memory use.

## Research directions and limits

This slice models independent binary route outcomes. It does not implement
contextual routing, delayed feedback, traffic-dependent effects, latency/cost
objectives, or live service dispatch. Fresh simulated hints reveal exact rates
and therefore are an intentionally privileged upper comparison. Stale and
misleading hints are controlled interventions, not measured advisor reliability.

Useful next questions are whether the gate still helps across unseen change
schedules, gradual drift, and variable hint error; whether evidence from one
route should invalidate an entire advisor; and whether a model that learns
changes explicitly improves over fixed forgetting. Evaluate on held-out scenarios
and multiple seeds before making comparative or novelty claims. A future
stateful environment could reuse the preserved MDP planner to study learning
transition models and replanning; the current one-step bandit interfaces do not
pretend to unify that problem already.

Thompson sampling and discounting historical evidence are established ideas; see
[A Tutorial on Thompson Sampling](https://arxiv.org/abs/1707.02038) and
[Taming Non-stationary Bandits: A Bayesian Approach](https://arxiv.org/abs/1707.09727).
The implementation here is not a claimed reproduction of the latter paper.

## Contents

| Area | Location | Includes |
| --- | --- | --- |
| Adaptive service routing | `src/bandits/adaptive.py` | changing reliability, imperfect advice, reusable agents, measured comparisons |
| Multi-armed bandits | `src/bandits` | epsilon-greedy, UCB, KL-UCB, and Thompson-sampling variants |
| Maze MDPs | `src/mdp` | grid encoder, value iteration, linear-programming and policy-iteration solvers, decoder, and visualizer |
| Windy Gridworld | `src/windy_gridworld` | Sarsa variants, Q-learning, and Expected Sarsa examples |

The supplied static data files are small course examples. Generated images, temporary solver files, and ad-hoc result paths are intentionally excluded from version control.

## Setup

The adaptive demo targets Python 3.11+ and uses the existing NumPy/Matplotlib
dependencies. Install the example dependencies in a Python environment:

```bash
python -m pip install -r requirements.txt
```

## Running an example

`run.py` is a small, portable launcher. It uses the Python interpreter that invoked it and locates its source files from the repository root, so it can be called from any working directory.

### Preserved bandits

The original `thompson-sampling-with-hint` code is retained. Its advice is generated
using the true maximum arm probability; it has privileged information and is
separate from the imperfect-hint policies in `adapt`. Historical reward/regret
semantics remain unchanged.


```bash
python run.py --verbose bandits \
  --instance data/bandits/instances/i-1.txt \
  --algorithm thompson-sampling \
  --rseed 42 \
  --epsilon 0.1 \
  --horizon 1000
```

### Maze MDP

```bash
python run.py --verbose solve_mdp \
  --grid data/mdp/grids/grid10.txt \
  --algorithm pi
```

Generate a fresh example instance when needed:

```bash
python run.py --verbose generate_mdp \
  --num_states 10 \
  --num_actions 5 \
  --gamma 0.95 \
  --mdptype episodic \
  --rseed 42 \
  --output_file src/mdp/tmp/generated_mdp.txt
```

Render a supplied grid or a path you have produced:

```bash
python run.py --verbose visualize_mdp \
  --grid_file data/mdp/grids/grid10.txt \
  --output_file plots/mdp/grid10.png
```

### Windy Gridworld

```bash
python run.py --verbose windy \
  --episodes 200 \
  --epsilon 0.15 \
  --discount 0.99 \
  --learning-rate 0.5
```

## Command reference

| Command | Description |
| --- | --- |
| `adapt` | Compare adaptive routing under drift and imperfect advice |
| `bandits` | Run a preserved multi-armed bandit example |
| `windy` | Run a Windy Gridworld example |
| `generate_mdp` | Generate a random MDP example |
| `solve_mdp` | Solve a maze MDP using value iteration, policy iteration, or linear programming |
| `visualize_mdp` | Render a maze grid or optional path |

## Implementation and checks

Maze state enumeration, action directions, and deterministic transitions are shared
in `src/mdp/core/grid.py`. The encoder still selects the nearest goal (file order
breaks ties); the decoder retains its historical last-goal behavior. Use a single
goal for the combined pipeline. Rewards, discount, and terminal handling are
unchanged. Solver temporary files occupy a private subdirectory under `src/mdp/tmp/`;
`--keep-tmp` prints its location. Existing files in the parent directory survive.

VI and PI need NumPy; LP additionally needs PuLP and its CBC solver. The documented
plot commands create their output directories. Expected Sarsa uses the weighted
sum of next-state Q values under the next epsilon-greedy policy.

A bounded smoke check is:

```bash
python run.py --verbose bandits --algorithm thompson-sampling-with-hint --horizon 100
python run.py --verbose solve_mdp --grid data/mdp/grids/grid10.txt --algorithm vi
```

The maze command opens a plot window and waits for it to close. For a headless
check, set `MPLBACKEND=Agg` in the command's environment.

The adaptive workflow has standard-library regression tests (no test dependency):

```bash
python -m unittest discover -v
python -m compileall -q run.py src tests
python -m black --check run.py src/bandits/adaptive.py src/bandits/environment.py src/bandits/experiment.py src/bandits/reporting.py src/bandits/scenario.py src/bandits/validation.py tests
```

Checks cover scenario diagnostics, numeric validation, workload bounds, paired
statistics, report-data round trips, safe HTML names, preservation of old reports,
failed-export cleanup, Unicode Windows output, and both command entrypoints.
Reporting is separate from simulation; `experiment.summarize`,
`experiment.paired_comparisons`, and `experiment.export_report` remain convenience
entrypoints. `export_report` requires a new destination directory and refuses to
replace an existing one. Larger streaming experiments and persisted online agent
state are future work.

## References

- [`references/mdp_references.txt`](references/mdp_references.txt)
- [`references/bandits_references.txt`](references/bandits_references.txt)
- [`references/windy_gridworld_references.txt`](references/windy_gridworld_references.txt)

## License

MIT. See [LICENSE](LICENSE).
