import math

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg

from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.observation_manager import (
    ObservationGroupCfg,
    ObservationTermCfg,
)
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg

from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.spec_config import GeomCfg
from mjlab.viewer import ViewerConfig

from robot_cfg import ROBOT_CFG, ACTUATED_JOINTS

from balance.rewards import (
    base_lin_vel_xy_l2,
    base_ang_vel_xy_l2,
    joint_soft_limit_penalty,
    support_contact_reward,
)


def balance_env_cfg() -> ManagerBasedRlEnvCfg:

    # ---------------------------------------------------------
    # Entity selectors
    # ---------------------------------------------------------

    robot_cfg = SceneEntityCfg(
        "robot",
        joint_names=ACTUATED_JOINTS,
        preserve_order=True,
    )

    limited_joint_cfg = SceneEntityCfg(
        "robot",
        joint_names=(
            *ACTUATED_JOINTS,
            "ankle_pitch_passive_4_left",
            "ankle_pitch_passive_4_right",
        ),
        preserve_order=True,
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
    # Terminations
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
    # Thiết kế riêng sau
    # ---------------------------------------------------------

    events = {}

    # ---------------------------------------------------------
    # Rewards
    # ---------------------------------------------------------

    rewards: dict[str, RewardTermCfg] = {

        # Robot còn sống, chưa vi phạm termination
        "alive": RewardTermCfg(
            func=mdp.is_alive,
            weight=1.0,
        ),

        # Giữ thân robot gần thẳng đứng
        "orientation": RewardTermCfg(
            func=mdp.flat_orientation_l2,
            weight=-2.0,
        ),

        # Giảm vận tốc tuyến tính của base trên mặt phẳng XY
        "base_lin_vel_xy": RewardTermCfg(
            func=base_lin_vel_xy_l2,
            weight=-1.0,
        ),

        # Giảm vận tốc góc roll/pitch của base
        "base_ang_vel_xy": RewardTermCfg(
            func=base_ang_vel_xy_l2,
            weight=-0.2,
        ),

        "joint_limit": RewardTermCfg(
            func=joint_soft_limit_penalty,
            weight=-0.1,
            params={
                "asset_cfg": limited_joint_cfg,
                "soft_ratio": 0.8,
            },
        ),

        "action_rate": RewardTermCfg(
            func=mdp.action_rate_l2,
            weight=-0.02,
        ),

        "support_contact": RewardTermCfg(
            func=support_contact_reward,
            weight=0.5,
            params={
                "min_normal_force": 1.0,
            },
        ),
    }

    # ---------------------------------------------------------
    # Terrain
    # ---------------------------------------------------------

    terrain = TerrainEntityCfg(
        terrain_type="plane",
        env_spacing=2.0,

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
    # Environment
    # ---------------------------------------------------------

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
        rewards=rewards,
        terminations=terminations,
        events=events,

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