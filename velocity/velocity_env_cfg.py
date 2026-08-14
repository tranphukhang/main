import math

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.tasks.velocity.mdp import (
    UniformVelocityCommandCfg,
    track_linear_velocity,
    track_angular_velocity,
    variable_posture,
    feet_slip,
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
from velocity.gait import feet_air_time_positive_biped


def velocity_env_cfg() -> ManagerBasedRlEnvCfg:

    # ---------------------------------------------------------
    # Entity selectors
    # ---------------------------------------------------------

    robot_cfg = SceneEntityCfg(
        "robot",
        joint_names=ACTUATED_JOINTS,
        preserve_order=True,
    )
    passive_ankle_cfg = SceneEntityCfg(
        "robot",
        joint_names=(
            "ankle_pitch_passive_4_left",
            "ankle_pitch_passive_4_right",
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

    toe_contact_cfg = ContactSensorCfg(
        name="toe_ground_contact",

        primary=ContactMatch(
            mode="geom",
            pattern=(
                r"left_foot_toe",
                r"right_foot_toe",
            ),
            entity="robot",
        ),

        secondary=ContactMatch(
            mode="body",
            pattern="terrain",
        ),

        fields=("found", "force"),
        reduce="netforce",
        num_slots=1,
    )

    heel_contact_cfg = ContactSensorCfg(
        name="heel_ground_contact",

        primary=ContactMatch(
            mode="geom",
            pattern=(
                r"left_foot_heel",
                r"right_foot_heel",
            ),
            entity="robot",
        ),

        secondary=ContactMatch(
            mode="body",
            pattern="terrain",
        ),

        fields=("found", "force"),
        reduce="netforce",
        num_slots=1,
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
                lin_vel_x=(0.1, 0.2),
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
            weight=1.0,
            params={
                "command_name": "twist",
                "std": 0.5,
            },
        ),

        "track_angular_velocity": RewardTermCfg(
            func=track_angular_velocity,
            weight=1.0,
            params={
                "command_name": "twist",
                "std": 0.5,
            },
        ),

        "orientation": RewardTermCfg(
            func=mdp.flat_orientation_l2,
            weight=-2.5,
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
            weight=-0.008,
        ),

        "feet_air_time": RewardTermCfg(
            func=feet_air_time_positive_biped,
            weight=0.25,
            params={
                "threshold": 0.8,
                "command_name": "twist",
                "command_threshold": 0.05,
                "sensor_name": "feet_ground_contact",
            },
        ),

        "feet_slip": RewardTermCfg(
            func=feet_slip,
            weight=-0.25,
            params={
                "sensor_name": "feet_ground_contact",
                "command_name": "twist",
                "command_threshold": 0.05,
                "asset_cfg": feet_cfg,
            },
        ),

        "ankle_mechanical_limit": RewardTermCfg(
            func=ankle_passive_soft_limit_penalty,
            weight=-0.25,
            params={
                "asset_cfg": passive_ankle_cfg,
                "soft_limit": math.radians(24.0),
                "hard_limit": math.radians(30.0),
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

            sensors=(
                feet_ground_cfg,
                toe_contact_cfg,
                heel_contact_cfg,
            ),

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