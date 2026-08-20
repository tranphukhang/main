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
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg

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
    support_contact_substep,
    support_contact_reward,
)
from balance.curriculums import push_force_curriculum


def balance_env_cfg() -> ManagerBasedRlEnvCfg:

    # ---------------------------------------------------------
    # Entity selectors
    # ---------------------------------------------------------

    robot_cfg = SceneEntityCfg(
        "robot",
        joint_names=ACTUATED_JOINTS,
        preserve_order=True,
    )

    base_body_cfg = SceneEntityCfg(
        "robot",
        body_names=("base",),
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
    # ---------------------------------------------------------

    events = {
        "reset_scene_to_default": EventTermCfg(
            func=mdp.reset_scene_to_default,
            mode="reset",
        ),

        "body_impulse": EventTermCfg(
            func=mdp.apply_body_impulse,
            mode="step",
            params={
                "force_range": (-3.0, 3.0),
                "torque_range": (0.0, 0.0),
                "duration_s": (0.1, 0.2),
                "cooldown_s": (2.0, 4.0),
                "asset_cfg": base_body_cfg,
            },
        ),
    }

    # ---------------------------------------------------------
    # Curriculum
    # ---------------------------------------------------------

    curriculum = {
        "push_force": CurriculumTermCfg(
            func=push_force_curriculum,
            params={
                "event_name": "body_impulse",

                "stages": [
                    {
                        "step": 0,
                        "max_force": 3.0,
                    },
                    {
                        "step": 100 * 24,
                        "max_force": 6.0,
                    },
                    {
                        "step": 200 * 24,
                        "max_force": 9.0,
                    },
                    {
                        "step": 300 * 24,
                        "max_force": 12.0,
                    },
                    {
                        "step": 400 * 24,
                        "max_force": 15.0,
                    },
                ],
            },
        ),

        "action_rate_weight": CurriculumTermCfg(
            func=mdp.reward_curriculum,
            params={
                "reward_name": "action_rate",

                "stages": [
                    {
                        "step": 0,
                        "weight": -0.03,
                    },
                    {
                        "step": 100 * 24,
                        "weight": -0.05,
                    },
                    {
                        "step": 200 * 24,
                        "weight": -0.07,
                    },
                    {
                        "step": 300 * 24,
                        "weight": -0.09,
                    },
                    {
                        "step": 400 * 24,
                        "weight": -0.10,
                    },
                ],
            },
        ),
    }

    # ---------------------------------------------------------
    # Metrics
    # ---------------------------------------------------------

    metrics = {
        "support_contact_substep": MetricsTermCfg(
            func=support_contact_substep,
            params={
                "min_normal_force": 1.0,
            },
            per_substep=True,
            reduce="mean",
        ),
    }

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
            weight=-0.03,
        ),

        "support_contact": RewardTermCfg(
            func=support_contact_reward,
            weight=0.5,
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
        curriculum=curriculum,
        metrics=metrics,

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