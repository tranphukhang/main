from pathlib import Path

import mujoco

from mjlab.actuator import DcMotorActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg


_ROOT_DIR = Path(__file__).parent
_ROBOT_XML = _ROOT_DIR / "robot.xml"


# 8 joints được điều khiển chủ động.
ACTUATED_JOINTS = (
    "hip_roll_left",
    "hip_pitch_left",
    "ankle_pitch_left",
    "calf_pitch_left",
    "hip_roll_right",
    "hip_pitch_right",
    "ankle_pitch_right",
    "calf_pitch_right",
)


def _get_robot_spec() -> mujoco.MjSpec:
    return mujoco.MjSpec.from_file(str(_ROBOT_XML))


ROBOT_ARTICULATION = EntityArticulationInfoCfg(
    actuators=(
        # RobStride 00
        # hip_roll + ankle_pitch
        DcMotorActuatorCfg(
            target_names_expr=(
                "hip_roll_left",
                "hip_roll_right",
                "ankle_pitch_left",
                "ankle_pitch_right",
            ),
            stiffness=100.0,
            damping=1.5,
            effort_limit=14.0,
            saturation_effort=14.0,
            velocity_limit=32.99,
        ),

        # RobStride 02
        # hip_pitch
        DcMotorActuatorCfg(
            target_names_expr=(
                "hip_pitch_left",
                "hip_pitch_right",
            ),
            stiffness=100.0,
            damping=1.5,
            effort_limit=17.0,
            saturation_effort=17.0,
            velocity_limit=42.94,
        ),

        # RobStride 02 + external transmission 1.5:1
        # calf_pitch
        DcMotorActuatorCfg(
            target_names_expr=(
                "calf_pitch_left",
                "calf_pitch_right",
            ),
            stiffness=100.0,
            damping=1.5,
            effort_limit=25.5,
            saturation_effort=25.5,
            velocity_limit=28.63,
        ),
    ),
)


ROBOT_CFG = EntityCfg(
    spec_fn=_get_robot_spec,
    articulation=ROBOT_ARTICULATION,
    init_state=EntityCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.3418),
        rot=(1.0, 0.0, 0.0, 0.0),

        # V0: chưa random gì hết.
        joint_pos={".*": 0.0},
        joint_vel={".*": 0.0},

        lin_vel=(0.0, 0.0, 0.0),
        ang_vel=(0.0, 0.0, 0.0),
    ),
)