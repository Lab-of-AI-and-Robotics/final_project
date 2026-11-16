# RL final project: Go2 locomotion (2026)

This course project uses Isaac Lab 3.0.0, Isaac Sim 6.2.0.0, Python 3.12,
and a sibling editable checkout of rsl_rl 5.5.1.

The course task remains `Template-Final-Project-v0`: Go2 velocity tracking on
flat terrain. Its environment, rewards, reset settings, and PPO hyperparameters
match the official Go2 Flat baseline in pinned Isaac Lab 3.0.0, with Isaac Sim
PhysX selected explicitly. The editable course configuration files remain in
this repository. Log names remain course-specific, and checkpoints use the
default RSL-RL behavior with a 50-iteration save interval.

## Install into the existing Isaac Lab environment

Expected sibling directories:

```text
rl_final_project_isaaclab_2026/
  IsaacLab/       # v3.0.0, with its .venv
  rsl_rl/         # v5.5.1, editable
  final_project/  # this repository
```

After installing Isaac Lab with its `isaacsim` extra, run:

```bash
cd /path/to/rl_final_project_isaaclab_2026/IsaacLab
uv add --editable ../rsl_rl --no-sync
uv add --editable ../final_project/source/final_project --no-sync
uv sync --extra isaacsim

uv run --extra isaacsim --no-sync python ../final_project/scripts/list_envs.py
```

This records the course package in Isaac Lab's project dependencies and installs
it into the same `.venv`. Continue to use `uv run` for the commands below.

Isaac Lab installs `rsl-rl-lib` as a dependency. The editable source declaration
above selects the sibling checkout instead of the packaged distribution.
Verify that the environment uses that checkout:

```bash
uv run --extra isaacsim --no-sync python -c \
  "import sys, rsl_rl; print(sys.executable); print(rsl_rl.__file__)"
```

## Check Go2 training and playback

Run these commands from the `IsaacLab` directory. Logs and checkpoints are saved
under `logs/rsl_rl/go2_managerbased/` relative to the working directory.

Start with a short headless check:

```bash
uv run --extra isaacsim --no-sync python ../final_project/scripts/rsl_rl/train.py \
  --task Template-Final-Project-v0 --num_envs 16 --max_iterations 10 \
  --viz none physics=isaacsim_physx
```

For the same check with a window, change `--viz none` to `--viz kit`.
The 10-iteration run checks execution; it does not establish policy convergence.

The course training defaults are 4096 environments and 300 iterations. Choose
`--num_envs` to fit the machine.

```bash
uv run --extra isaacsim --no-sync python ../final_project/scripts/rsl_rl/train.py \
  --task Template-Final-Project-v0 --viz none physics=isaacsim_physx
```

Play a checkpoint produced by this environment:

```bash
uv run --extra isaacsim --no-sync python ../final_project/scripts/rsl_rl/play.py \
  --task Template-Final-Project-Play-v0 --num_envs 16 \
  --checkpoint /absolute/path/to/model_299.pt \
  --viz kit physics=isaacsim_physx
```

The scripts delegate training, playback, and dummy-agent execution to Isaac Lab 3.
Task registration also supports the unified `isaaclab train` and `isaaclab play`
commands once the package is installed. The `isaaclab.tasks` entry point makes
course tasks discoverable without editing Isaac Lab's source.
