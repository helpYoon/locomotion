"""Stand-in for booster_deploy.robots.t2: T2_31DOF_CFG with the vendor's joint order, effort limits and prepare state
(copied; tests compare them with the vendor file). Its own gains and default pose are placeholders: the task replaces
them, and a test would catch a task that did not."""

from ..controllers.controller_cfg import PrepareStateCfg, RobotCfg

T2_31DOF_CFG = RobotCfg(
    name="Booster_T2_31DOF",
    joint_names=['aa_head_yaw_joint', 'head_pitch_joint', 'left_shoulder_pitch_joint', 'left_shoulder_roll_joint',
 'left_elbow_pitch_joint', 'left_elbow_yaw_joint', 'left_wrist_pitch_joint', 'left_wrist_yaw_joint',
 'left_wrist_roll_joint', 'right_shoulder_pitch_joint', 'right_shoulder_roll_joint', 'right_elbow_pitch_joint',
 'right_elbow_yaw_joint', 'right_wrist_pitch_joint', 'right_wrist_yaw_joint', 'right_wrist_roll_joint',
 'waist_pitch_joint', 'waist_roll_joint', 'waist_yaw_joint', 'left_hip_pitch_joint', 'left_hip_roll_joint',
 'left_hip_yaw_joint', 'left_knee_pitch_joint', 'left_ankle_pitch_joint', 'left_ankle_roll_joint',
 'right_hip_pitch_joint', 'right_hip_roll_joint', 'right_hip_yaw_joint', 'right_knee_pitch_joint',
 'right_ankle_pitch_joint', 'right_ankle_roll_joint'],
    joint_stiffness=[1.0] * 31,
    joint_damping=[1.0] * 31,
    default_joint_pos=[9.0] * 31,
    effort_limit=[22.0, 22.0, 74.0, 60.0, 60.0, 60.0, 22.0, 22.0, 22.0, 74.0, 60.0, 60.0, 60.0, 22.0, 22.0, 22.0, 138.0, 144.0, 135.0, 135.0, 135.0, 135.0, 135.0, 132.0, 72.0, 135.0, 135.0, 135.0, 135.0, 132.0, 72.0],
    mjcf_path='{BOOSTER_ASSETS_DIR}/robots/T2/T2_31dof/T2_31dof.xml',
    prepare_state=PrepareStateCfg(
        stiffness=[40.0, 40.0, 70.0, 70.0, 70.0, 70.0, 55.0, 70.0, 55.0, 70.0, 70.0, 70.0, 70.0, 55.0, 70.0, 55.0, 500.0, 500.0, 400.0, 500.0, 350.0, 180.0, 500.0, 500.0, 500.0, 500.0, 350.0, 180.0, 500.0, 500.0, 500.0],
        damping=[0.65, 0.65, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0],
        joint_pos=[0.0, 0.0, 0.2, -1.3, 0.0, -0.5, 0.0, 0.0, 0.0, 0.2, 1.3, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.0, 0.0, 0.0, 0.23, -0.18, 0.0, -0.0, -0.0, 0.0, 0.23, -0.18, 0.0],
    ),
)
