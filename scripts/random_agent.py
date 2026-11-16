# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Run a course task with random actions."""

import sys

import final_project.tasks  # noqa: F401
from isaaclab.cli import random_agent

if __name__ == "__main__":
    random_agent(sys.argv[1:])
