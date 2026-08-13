import math

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.tasks.velocity.mdp import (
    UniformVelocityCommandCfg,
    track_linear_velocity,
    track_angular_velocity,
    variable_posture,
)

from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.observation_manager import (
    ObservationGroupCfg,
    ObservationTermCfg,
)
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.command_manager import CommandTermCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg

from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.spec_config import GeomCfg
from mjlab.viewer import ViewerConfig
from mjlab.sensor import ContactMatch, ContactSensorCfg

from robot_cfg import ROBOT_CFG, ACTUATED_JOINTS

from standing.rewards import ankle_mechanical_limit_penalty
from velocity.gait import gait_phase, feet_gait, feet_clearance_flat


def velocity_env_cfg() -> ManagerBasedRlEnvCfg:

    # ---------------------------------------------------------
    # Entity selectors
    # ---------------------------------------------------------

    robot_cfg = SceneEntityCfg(
        "robot",
        joint_names=ACTUATED_JOINTS,
        preserve_order=True,
    )

    ankle_mechanical_cfg = SceneEntityCfg(
        "robot",
        joint_names=(
            "ankle_pitch_left",
            "calf_pitch_left",
            "ankle_pitch_right",
            "calf_pitch_right",
        ),
        preserve_order=True,
    )

    feet_cfg = SceneEntityCfg(
        "robot",
        site_names=(
            "left_foot",
            "right_foot",
        ),
        preserve_order=True,
    )

    feet_ground_cfg = ContactSensorCfg(
        name="feet_ground_contact",

        primary=ContactMatch(
            mode="subtree",
            pattern=r"^(feet_left|feet_right)$",
            entity="robot",
        ),

        secondary=ContactMatch(
            mode="body",
            pattern="terrain",
        ),

        fields=("found", "force"),
        reduce="netforce",
        num_slots=1,

        track_air_time=True,
    )


    # ---------------------------------------------------------
    # Observations
    # ---------------------------------------------------------

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

        "phase": ObservationTermCfg(
            func=gait_phase,
            params={
                "period": 2.0,
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

    # ---------------------------------------------------------
    # Actions
    # ---------------------------------------------------------

    actions: dict[str, ActionTermCfg] = {
        "joint_pos": JointPositionActionCfg(
            entity_name="robot",
            actuator_names=ACTUATED_JOINTS,
            scale=0.5,
            use_default_offset=True,
        ),
    }

    # ---------------------------------------------------------
    # Commands
    # ---------------------------------------------------------

    commands: dict[str, CommandTermCfg] = {
        "twist": UniformVelocityCommandCfg(
            entity_name="robot",

            resampling_time_range=(3.0, 5.0),

            rel_standing_envs=0.1,
            rel_heading_envs=0.0,
            rel_world_envs=0.0,
            rel_forward_envs=0.0,

            heading_command=False,

            debug_vis=True,

            ranges=UniformVelocityCommandCfg.Ranges(
                lin_vel_x=(0.05, 0.2),
                lin_vel_y=(0.0, 0.0),
                ang_vel_z=(0.0, 0.0),
            ),
        ),
    }

    # ---------------------------------------------------------
    # Termination
    # ---------------------------------------------------------

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

    # ---------------------------------------------------------
    # Events
    # ---------------------------------------------------------

    events = {
        "reset_scene_to_default": EventTermCfg(
            func=mdp.reset_scene_to_default,
            mode="reset",
        ),
    }

    # ---------------------------------------------------------
    # Rewards
    # ---------------------------------------------------------

    rewards: dict[str, RewardTermCfg] = {
        "track_linear_velocity": RewardTermCfg(
            func=track_linear_velocity,
            weight=2.0,
            params={
                "command_name": "twist",
                "std": 0.30,
            },
        ),

        "track_angular_velocity": RewardTermCfg(
            func=track_angular_velocity,
            weight=1.5,
            params={
                "command_name": "twist",
                "std": 0.4,
            },
        ),

        "orientation": RewardTermCfg(
            func=mdp.flat_orientation_l2,
            weight=-1.0,
        ),

        "joint_velocity": RewardTermCfg(
            func=mdp.joint_vel_l2,
            weight=-0.0005,
            params={
                "asset_cfg": robot_cfg,
            },
        ),

        "posture": RewardTermCfg(
            func=variable_posture,
            weight=1.0,
            params={
                "asset_cfg": robot_cfg,
                "command_name": "twist",

                "std_standing": {
                    ".*": 0.05,
                },

                "std_walking": {
                    r"hip_roll_.*": 0.08,

                    r"hip_pitch_.*": 0.45,

                    r"calf_pitch_.*": 0.45,

                    r"ankle_pitch_.*": 0.20,
                },

                "std_running": {
                    ".*": 0.30,
                },

                "walking_threshold": 0.05,
                "running_threshold": 1.5,
            },
        ),

        "action_rate": RewardTermCfg(
            func=mdp.action_rate_l2,
            weight=-0.05,
        ),

        "ankle_mechanical_limit": RewardTermCfg(
            func=ankle_mechanical_limit_penalty,
            weight=-0.5,
            params={
                "asset_cfg": ankle_mechanical_cfg,
                "limit": math.radians(30.0),
            },
        ),

        "foot_gait": RewardTermCfg(
            func=feet_gait,
            weight=0.75,
            params={
                "period": 2.0,

                # Left / Right lệch nhau nửa chu kỳ.
                "offset": [0.0, 0.5],

                # 56% stance → có double-support ngắn.
                "threshold": 0.56,

                "command_threshold": 0.05,
                "command_name": "twist",
                "sensor_name": "feet_ground_contact",
            },
        ),

        "foot_clearance": RewardTermCfg(
            func=feet_clearance_flat,
            weight=-2.0,
            params={
                "target_height": 0.1,
                "command_name": "twist",
                "command_threshold": 0.05,
                "asset_cfg": feet_cfg,
            },
        ),
    }

    # ---------------------------------------------------------
    # Terrain
    # ---------------------------------------------------------

    terrain = TerrainEntityCfg(
        terrain_type="plane",
        env_spacing=2.0,

        # robot hiện tại sử dụng collision bit 2 và 4.
        # plane_ground_scene.xml dùng conaffinity="15".
        geoms=(
            GeomCfg(
                geom_names_expr=("terrain$",),
                contype=1,
                conaffinity=15,
                condim=3,
            ),
        ),
    )

    # ---------------------------------------------------------
    # Curriculum
    # ---------------------------------------------------------

    curriculum = {
        "orientation_weight": CurriculumTermCfg(
            func=mdp.reward_curriculum,
            params={
                "reward_name": "orientation",
                "stages": [
                    {"step": 0 * 24,    "weight": -0.50},
                    {"step": 500 * 24,  "weight": -1.00},
                    {"step": 1500 * 24, "weight": -1.50},
                    {"step": 2500 * 24, "weight": -2.00},
                    {"step": 3500 * 24, "weight": -3.00},
                    {"step": 4500 * 24, "weight": -4.00},
                ],
            },
        ),
    }

    # ---------------------------------------------------------
    # Environment
    # ---------------------------------------------------------

    return ManagerBasedRlEnvCfg(

        scene=SceneCfg(
            terrain=terrain,

            entities={
                "robot": ROBOT_CFG,
            },

            sensors=(feet_ground_cfg,),

            num_envs=1,
            env_spacing=0.5,
        ),

        observations=observations,
        actions=actions,
        commands=commands,
        rewards=rewards,
        terminations=terminations,
        events=events,
        curriculum=curriculum,

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

        # 0.0005 * 40 = 0.02 s
        # control frequency = 50 Hz
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