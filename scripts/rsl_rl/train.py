# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Train the course task through Isaac Lab 3."""

import sys

import final_project.tasks  # noqa: F401
from isaaclab.cli import train

if __name__ == "__main__":
    train(["--rl_library", "rsl_rl", *sys.argv[1:]])
