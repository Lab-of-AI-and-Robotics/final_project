# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""List course tasks without launching the simulator."""

import sys

import final_project.tasks  # noqa: F401
from isaaclab.cli import list_envs

if __name__ == "__main__":
    list_envs(["--keyword", "Template-Final-Project", *sys.argv[1:]])
