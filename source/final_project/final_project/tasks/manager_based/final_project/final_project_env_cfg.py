# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG
from isaaclab_physx.physics import PhysxCfg
from isaaclab_tasks.core.velocity.velocity_env_cfg import (
    LocomotionVelocityRoughEnvCfg,
    RewardsCfg,
)
from isaaclab_tasks.utils import PresetCfg

from . import mdp


@configclass
class FinalProjectRewardsCfg(RewardsCfg):
    """Course rewards matching the pinned official Go2 flat baseline."""

    base_height_l2 = RewTerm(
        func=mdp.base_height_l2,
        weight=-30.0,
        params={"target_height": 0.4, "sensor_cfg": None},
    )


@configclass
class FinalProjectPhysicsCfg(PresetCfg):
    """Use Isaac Sim PhysX for the course, including headless training."""

    isaacsim_physx = PhysxCfg(gpu_max_rigid_patch_count=10 * 2**15)
    default = isaacsim_physx


##
# Scene definition
##


@configclass
class FinalProjectEnvCfg(LocomotionVelocityRoughEnvCfg):
    sim: SimulationCfg = SimulationCfg(physics=FinalProjectPhysicsCfg())
    rewards: FinalProjectRewardsCfg = FinalProjectRewardsCfg()

    def __post_init__(self):
        # Call the parent's post_init method first
        super().__post_init__()

        # -- Scene Configuration
        # Use the same Go2 asset and motor path as the official flat task.
        self.sim.use_newton_actuators = True
        self.scene.robot = UNITREE_GO2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        # Change terrain to a flat plane for faster training
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        # Remove the height scanner since the terrain is flat
        self.scene.height_scanner = None
        # Also remove height scan from observations
        self.observations.policy.height_scan = None
        # Disable the terrain curriculum
        self.curriculum.terrain_levels = None

        # -- Action Configuration
        # Set the scale for joint position actions
        self.actions.joint_pos.scale = 0.25

        # -- Event Configuration
        # Configure the robot's initial state randomization
        self.events.reset_base.params = {
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (-0.5, 0.5),
                "roll": (-0.5, 0.5),
                "pitch": (-0.5, 0.5),
                "yaw": (-0.5, 0.5),
            },
        }
        self.events.reset_robot_joints.params["position_range"] = (0.5, 1.5)

        # -- Reward Configuration
        # These weights are crucial for learning to walk!
        self.rewards.track_lin_vel_xy_exp.weight = 1.5
        self.rewards.track_ang_vel_z_exp.weight = 0.75
        self.rewards.dof_torques_l2.weight = -0.0002
        self.rewards.feet_air_time.params["sensor_cfg"].body_names = ".*_foot"
        self.rewards.flat_orientation_l2.weight = -2.5
        self.rewards.feet_air_time.weight = 0.25
        self.rewards.action_rate_l2.weight = -0.005
        self.rewards.undesired_contacts = None

        # -- Termination Configuration
        # The episode ends if the robot's base hits the ground
        self.terminations.base_contact.params["sensor_cfg"].body_names = "base"


@configclass
class FinalProjectEnvCfg_PLAY(FinalProjectEnvCfg):
    def __post_init__(self):
        # First, inherit all settings from the training config
        super().__post_init__()

        # --- Settings to override for playing ---
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing event
        self.events.base_external_force_torque = None
        self.events.push_robot = None
