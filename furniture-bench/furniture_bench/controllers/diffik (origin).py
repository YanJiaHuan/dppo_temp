from typing import Dict

import torch

import furniture_bench.controllers.control_utils as C

import torch

from ipdb import set_trace as bp


# from Robotic_Arm.rm_robot_interface import *

def diffik_factory(real_robot=True, *args, **kwargs):
    if real_robot:
        import torchcontrol as toco

        base = toco.PolicyModule
    else:
        base = torch.nn.Module

    class DiffIKController(base):
        """Differential Inverse Kinematics Controller"""

        def __init__(
            self,
            pos_scalar=1.0,
            rot_scalar=1.0,
        ):
            """Initialize Differential Inverse Kinematics Controller.

            Args:
            """
            super().__init__()
            self.ee_pos_desired = None
            self.ee_quat_desired = None
            self.ee_pos_error = None
            self.ee_rot_error = None

            self.pos_scalar = pos_scalar
            self.rot_scalar = rot_scalar

            self.scale_errors = True

            # self.arm_model = rm_robot_arm_model_e.RM_MODEL_GEN_72_E  # gen_72机械臂
            # self.force_type = rm_force_type_e.RM_MODEL_RM_B_E
            # self.algo_handle = Algo(arm_model, force_type)

            print(
                f"Making DiffIK controller with pos_scalar: {pos_scalar}, rot_scalar: {rot_scalar}"
            )

        def forward(
            self, state_dict: Dict[str, torch.Tensor]
        ) -> Dict[str, torch.Tensor]:
            # Get states.
            # Shape of joint_pos_current: (batch_size, num_joints = 7)
            joint_pos_current = state_dict["joint_positions"]

            # Shape of jacobian: (batch_size, 6, num_joints = 7)
            jacobian = state_dict["jacobian_diffik"]

            # Shape of ee_pos: (batch_size, 3)
            # Shape of ee_quat: (batch_size, 4) with real part at the end
            ee_pos, ee_quat_xyzw = state_dict["ee_pos"], state_dict["ee_quat"]
            goal_ori_xyzw = self.goal_ori

            position_error = self.goal_pos - ee_pos

            # Convert quaternions to rotation matrices
            ee_mat = C.quaternion_to_matrix(ee_quat_xyzw)
            goal_mat = C.quaternion_to_matrix(goal_ori_xyzw)

            # Compute the matrix error
            mat_error = torch.matmul(goal_mat, torch.inverse(ee_mat))

            # Convert the matrix error to axis-angle representation
            ee_delta_axis_angle = C.matrix_to_axis_angle(mat_error)

            dt = 0.1

            ee_pos_vel = position_error * self.pos_scalar / dt
            ee_rot_vel = ee_delta_axis_angle * self.rot_scalar / dt

            ee_velocity_desired = torch.cat((ee_pos_vel, ee_rot_vel), dim=-1)
            joint_vel_desired = torch.linalg.lstsq(
                jacobian, ee_velocity_desired
            ).solution
            joint_pos_desired = joint_pos_current + joint_vel_desired * dt
            # joint_pos_desired = joint_pos_current
            # convert from tensor to np
            # joints = joint_pos_desired.detach().cpu().numpy()
            # target_pos = self.goal_pos.detach().cpu().numpy()
            # target_quat = self.goal_ori.detach().cpu().numpy()
            # target_euler = algo_handle.rm_algo_quaternion2euler(target_quat.tolist())
            # target_pose = target_pos.tolist() + target_euler
            # params = rm_inverse_kinematics_params_t(joints.tolist(), target_pose, 1)
            # q_out = algo_handle.rm_algo_inverse_kinematics(params)

            return {"joint_positions": joint_pos_desired}

        def set_goal(self, goal_pos, goal_ori):
            self.goal_pos = goal_pos
            self.goal_ori = goal_ori

        def reset(self):
            pass

    return DiffIKController(*args, **kwargs)
