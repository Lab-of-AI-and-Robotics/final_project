# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Play a course checkpoint through Isaac Lab 3."""

import sys

import final_project.tasks  # noqa: F401
from isaaclab.cli import play

if __name__ == "__main__":
    play(["--rl_library", "rsl_rl", *sys.argv[1:]])
