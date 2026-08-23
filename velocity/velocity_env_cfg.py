import math

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp
from mjlab.envs.mdp.actions import (
    JointPositionActionCfg,
)
from mjlab.tasks.velocity.mdp import (
    UniformVelocityCommandCfg,
    track_angular_velocity,
    track_linear_velocity,
)

from mjlab.managers.action_manager import (
    ActionTermCfg,
)
from mjlab.managers.command_manager import (
    CommandTermCfg,
)
from mjlab.managers.event_manager import (
    EventTermCfg,
)
from mjlab.managers.observation_manager import (
    ObservationGroupCfg,
    ObservationTermCfg,
)
from mjlab.managers.reward_manager import (
    RewardTermCfg,
)
from mjlab.managers.scene_entity_config import (
    SceneEntityCfg,
)
from mjlab.managers.termination_manager import (
    TerminationTermCfg,
)

from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.spec_config import GeomCfg
from mjlab.viewer import ViewerConfig

from robot_cfg import ACTUATED_JOINTS, ROBOT_CFG


def velocity_env_cfg() -> ManagerBasedRlEnvCfg:

    # =========================================================
    # Entity selectors
    # =========================================================

    robot_cfg = SceneEntityCfg(
        "robot",
        joint_names=ACTUATED_JOINTS,
        preserve_order=True,
    )

    # =========================================================
    # Observations
    # =========================================================

    actor_terms = {
        "base_lin_vel": ObservationTermCfg(
            func=mdp.base_lin_vel,
        ),
        "base_ang_vel": ObservationTermCfg(
            func=mdp.base_ang_vel,
        ),
        "projected_gravity": ObservationTermCfg(
            func=mdp.projected_gravity,
        ),
        "joint_pos": ObservationTermCfg(
            func=mdp.joint_pos_rel,
            params={
                "asset_cfg": robot_cfg,
            },
        ),
        "joint_vel": ObservationTermCfg(
            func=mdp.joint_vel_rel,
            params={
                "asset_cfg": robot_cfg,
            },
        ),
        "actions": ObservationTermCfg(
            func=mdp.last_action,
        ),
        "command": ObservationTermCfg(
            func=mdp.generated_commands,
            params={
                "command_name": "twist",
            },
        ),
    }

    observations = {
        "actor": ObservationGroupCfg(
            terms=actor_terms,
            concatenate_terms=True,
            enable_corruption=False,
        ),
        "critic": ObservationGroupCfg(
            terms={**actor_terms},
            concatenate_terms=True,
            enable_corruption=False,
        ),
    }

    # =========================================================
    # Actions
    # =========================================================

    actions: dict[str, ActionTermCfg] = {
        "joint_pos": JointPositionActionCfg(
            entity_name="robot",
            actuator_names=ACTUATED_JOINTS,
            scale=0.5,
            use_default_offset=True,
        ),
    }

    # =========================================================
    # Velocity command
    # =========================================================

    commands: dict[str, CommandTermCfg] = {
        "twist": UniformVelocityCommandCfg(
            entity_name="robot",

            resampling_time_range=(4.0, 6.0),

            # 10% môi trường được yêu cầu đứng yên.
            rel_standing_envs=0.1,

            rel_heading_envs=0.0,
            rel_world_envs=0.0,
            rel_forward_envs=0.0,

            heading_command=False,
            debug_vis=True,

            # Phase 1: chỉ học bước tiến.
            ranges=UniformVelocityCommandCfg.Ranges(
                lin_vel_x=(0.05, 0.25),
                lin_vel_y=(0.0, 0.0),
                ang_vel_z=(0.0, 0.0),
            ),
        ),
    }

    # =========================================================
    # Terminations
    # =========================================================

    terminations = {
        "time_out": TerminationTermCfg(
            func=mdp.time_out,
            time_out=True,
        ),
        "bad_orientation": TerminationTermCfg(
            func=mdp.bad_orientation,
            params={
                "limit_angle": math.radians(60.0),
            },
        ),
    }

    # =========================================================
    # Reset event
    # =========================================================

    events = {
        "reset_scene_to_default": EventTermCfg(
            func=mdp.reset_scene_to_default,
            mode="reset",
        ),
    }

    # =========================================================
    # Initial rewards
    # =========================================================

    rewards: dict[str, RewardTermCfg] = {
        "alive": RewardTermCfg(
            func=mdp.is_alive,
            weight=1.0,
        ),
        "termination_penalty": RewardTermCfg(
            func=mdp.is_terminated,
            weight=-100.0,
        ),
        "track_linear_velocity": RewardTermCfg(
            func=track_linear_velocity,
            weight=1.0,
            params={
                "command_name": "twist",
                "std": 0.2,
            },
        ),
        "track_angular_velocity": RewardTermCfg(
            func=track_angular_velocity,
            weight=0.5,
            params={
                "command_name": "twist",
                "std": 0.5,
            },
        ),
        "orientation": RewardTermCfg(
            func=mdp.flat_orientation_l2,
            weight=-2.0,
        ),
        "joint_acceleration": RewardTermCfg(
            func=mdp.joint_acc_l2,
            weight=-2.0e-7,
            params={
                "asset_cfg": robot_cfg,
            },
        ),
        "joint_torque": RewardTermCfg(
            func=mdp.joint_torques_l2,
            weight=-1.0e-6,
        ),
        "action_rate": RewardTermCfg(
            func=mdp.action_rate_l2,
            weight=-0.02,
        ),
    }

    # =========================================================
    # Terrain
    # =========================================================

    terrain = TerrainEntityCfg(
        terrain_type="plane",
        env_spacing=2.0,
        geoms=(
            GeomCfg(
                geom_names_expr=(
                    "terrain$",
                ),
                contype=1,
                conaffinity=15,
                condim=3,
            ),
        ),
    )

    # =========================================================
    # Environment
    # =========================================================

    return ManagerBasedRlEnvCfg(
        scene=SceneCfg(
            terrain=terrain,
            entities={
                "robot": ROBOT_CFG,
            },
            num_envs=1,
            env_spacing=0.5,
        ),
        observations=observations,
        actions=actions,
        commands=commands,
        rewards=rewards,
        terminations=terminations,
        events=events,
        curriculum={},
        sim=SimulationCfg(
            mujoco=MujocoCfg(
                timestep=0.0005,
                integrator="implicitfast",
                solver="newton",
                cone="elliptic",
                tolerance=1e-10,
                disableflags=("nativeccd",),
            ),
        ),

        # 0.0005 × 40 = 0.02 s → 50 Hz
        decimation=40,
        episode_length_s=10.0,
        viewer=ViewerConfig(
            origin_type=ViewerConfig.OriginType.ASSET_ROOT,
            entity_name="robot",
            distance=1.5,
            elevation=-10.0,
            azimuth=90.0,
        ),
    )