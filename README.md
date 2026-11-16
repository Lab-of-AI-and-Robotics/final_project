# RL final project Go2 locomotion 2026

## 1. Project Setup

This guide installs Isaac Sim, Isaac Lab, a separate editable copy of rsl_rl,
and the course project on Ubuntu Linux. The course uses Go2 velocity tracking
on flat terrain with Isaac Sim PhysX and the official Go2 Flat settings.

Use the following versions for this course:

| Component | Version |
| --- | --- |
| Python | 3.12.16 |
| Isaac Lab | v3.0.0 |
| Isaac Sim | 6.2.0.0, installed through the `isaacsim` extra |
| rsl_rl | v5.5.1 |
| Course project | `course-2026` branch |

### 1.1. Installation with uv

Official references: [Isaac Lab installation](https://isaac-sim.github.io/IsaacLab/main/source/isaaclab/installation.html)
and [uv installation](https://docs.astral.sh/uv/getting-started/installation/).
Check the official hardware and driver requirements before installation.

#### 1.1.1. Prerequisites and uv

Install the system tools:

```bash
sudo apt update
sudo apt install git curl cmake build-essential
```

Check that the NVIDIA GPU and driver are detected:

```bash
nvidia-smi
```

Install uv:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source "$HOME/.local/bin/env"
uv --version
```

If uv was already installed with the standalone installer, update it with
`uv self update` instead.

Conda is not required. uv creates a project environment in `IsaacLab/.venv`.
The commands below use that environment through `uv run`.

#### 1.1.2. Project directory and Isaac Lab

Choose an SSD with enough free space for Isaac Sim, the Python environment,
and download caches. The example project root is `~/Projects/rl_final_project_isaaclab_2026`.

```bash
mkdir -p ~/Projects/rl_final_project_isaaclab_2026
cd ~/Projects/rl_final_project_isaaclab_2026

git clone --branch v3.0.0 --depth 1 \
  https://github.com/isaac-sim/IsaacLab.git
```

#### 1.1.3. Editable rsl_rl installation and Isaac Sim

Clone the course version of rsl_rl into a separate sibling directory:

```bash
cd ~/Projects/rl_final_project_isaaclab_2026

git clone --branch v5.5.1 --depth 1 \
  https://github.com/leggedrobotics/rsl_rl.git

git -C rsl_rl switch -c course-2026
```

Register that checkout as an editable dependency and install the environment:

```bash
cd ~/Projects/rl_final_project_isaaclab_2026/IsaacLab

uv python install 3.12.16
uv python pin 3.12.16
uv add --editable ../rsl_rl --no-sync
uv sync --extra isaacsim
```

The `isaacsim` extra installs Isaac Sim and its required packages. A separate
manual Isaac Sim or PyTorch pip installation is not needed for this setup.
Isaac Lab already depends on rsl_rl; the editable dependency above selects our
local checkout as the installed source. No preliminary `pip uninstall` is needed.

Verify the Python environment and rsl_rl source location:

```bash
uv run --extra isaacsim --no-sync python -c \
  "import sys, rsl_rl; print(sys.executable); print(rsl_rl.__file__)"
```

The paths should end with:

```text
rl_final_project_isaaclab_2026/IsaacLab/.venv/bin/python
rl_final_project_isaaclab_2026/rsl_rl/rsl_rl/__init__.py
```

#### 1.1.4. Verify the simulator

Run the empty-scene tutorial from the `IsaacLab` directory:

```bash
uv run --extra isaacsim --no-sync python \
  scripts/tutorials/00_sim/create_empty.py --viz kit
```

Accept the NVIDIA EULA if prompted. Wait for initialization to finish, then
stop the example with Ctrl+C in the terminal.

#### 1.1.5. Install the final project

Clone the course branch:

```bash
cd ~/Projects/rl_final_project_isaaclab_2026

git clone --branch course-2026 \
  https://github.com/Lab-of-AI-and-Robotics/final_project.git
```

Install it into the Isaac Lab environment:

```bash
cd ~/Projects/rl_final_project_isaaclab_2026/IsaacLab

uv add --editable ../final_project/source/final_project --no-sync
uv sync --extra isaacsim

uv run --extra isaacsim --no-sync python ../final_project/scripts/list_envs.py
```

The folder structure is:

```text
rl_final_project_isaaclab_2026/
  IsaacLab/
    .venv/
  rsl_rl/
  final_project/
```

The environment list should include `Template-Final-Project-v0` and
`Template-Final-Project-Play-v0`. These are the training and playback tasks
used for this course.

### 1.2. Training and Inference

Run all commands in this section from:

```bash
cd ~/Projects/rl_final_project_isaaclab_2026/IsaacLab
```

#### 1.2.1. Short installation check

Start with 16 environments and 10 training iterations:

```bash
uv run --extra isaacsim --no-sync python ../final_project/scripts/rsl_rl/train.py \
  --task Template-Final-Project-v0 \
  --num_envs 16 --max_iterations 10 \
  --viz none physics=isaacsim_physx
```

`--viz none` runs without a window. Change it to `--viz kit` to display the
simulation. This short run checks that the installation works.

#### 1.2.2. Training

The default training uses 4096 parallel environments and 300 iterations.
Reduce `--num_envs` if needed for your GPU memory.

```bash
uv run --extra isaacsim --no-sync python ../final_project/scripts/rsl_rl/train.py \
  --task Template-Final-Project-v0 \
  --viz none physics=isaacsim_physx
```

Training logs and checkpoints are saved under
`logs/rsl_rl/go2_managerbased/<RUN_FOLDER>/`. Checkpoints use the standard
rsl_rl saving behavior with a 50-iteration interval. A completed 300-iteration
run produces the final checkpoint `model_299.pt`.

#### 1.2.3. Inference

Replace the checkpoint path below with the path from your training run:

```bash
uv run --extra isaacsim --no-sync python ../final_project/scripts/rsl_rl/play.py \
  --task Template-Final-Project-Play-v0 \
  --checkpoint /absolute/path/to/model_299.pt \
  --num_envs 1 --viz kit --real-time \
  physics=isaacsim_physx \
  'env.viewer.eye=[3.0,3.0,2.0]' \
  'env.viewer.lookat=[0.0,0.0,0.0]' \
  env.viewer.origin_type=asset_root \
  env.viewer.asset_name=robot
```

The camera follows the robot. Press Ctrl+C in the terminal to stop playback.

### 1.3. Editing PPO

The PPO implementation is in the separate rsl_rl checkout:

```text
rl_final_project_isaaclab_2026/rsl_rl/rsl_rl/algorithms/ppo.py
```

The course's PPO configuration is located at:

```text
final_project/source/final_project/final_project/tasks/manager_based/final_project/agents/rsl_rl_ppo_cfg.py
```

Because rsl_rl is installed in editable mode, source changes take effect when
you start a new training process. You do not need to reinstall the package after
each source edit. An already-running process continues using the code it loaded
at startup.
