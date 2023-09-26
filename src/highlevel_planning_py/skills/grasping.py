import pybullet as p
import numpy as np
from scipy.spatial.transform import Rotation as R
from highlevel_planning_py.tools_pl.util import (
    SkillExecutionError,
    IKError,
    ConstraintSpec,
)


def get_object_link_pose(body_id, link_id):
    if link_id == -1:
        temp = p.getBasePositionAndOrientation(body_id)
        r_O_O_obj = np.array(temp[0]).reshape((-1, 1))
        C_O_obj = R.from_quat(np.array(temp[1]))
    else:
        temp = p.getLinkState(body_id, link_id)
        r_O_O_obj = np.array(temp[4]).reshape((-1, 1))
        C_O_obj = R.from_quat(np.array(temp[5]))
    return r_O_O_obj, C_O_obj


class SkillGrasping:
    def __init__(self, scene_, robot_, config):
        self.scene = scene_
        self.robot = robot_

        self.last_pre_pos = None
        self.last_pre_orient = None

        self._pregrasp_z_offset = config.getparam(
            ["grasping", "pregrasp_z_offset"], default_value=0.15
        )

    def compute_grasp(self, target_name, link_idx=0, grasp_id=0):
        obj_info = self.scene.objects[target_name]
        target_id = obj_info.model.uid
        if len(obj_info.grasp_links) == 0:
            raise SkillExecutionError("No grasps defined for this object")
        link_id = obj_info.grasp_links[link_idx]

        num_grasps = len(obj_info.grasp_pos[link_id])
        if num_grasps == 0:
            raise SkillExecutionError("No grasps defined for this object")
        if grasp_id >= num_grasps:
            raise SkillExecutionError("Invalid grasp ID")

        # Get the object pose
        r_O_O_obj, C_O_obj = get_object_link_pose(target_id, link_id)

        # Get grasp data
        r_Obj_obj_grasp = obj_info.grasp_pos[link_id][grasp_id].reshape((-1, 1))

        # Get robot arm base orientation
        temp1 = p.getLinkState(self.robot.model.uid, self.robot.arm_base_link_idx)
        C_O_rob = R.from_quat(np.array(temp1[5]))

        # Compute desired position of end effector in robot frame
        r_O_O_grasp = r_O_O_obj + C_O_obj.apply(r_Obj_obj_grasp.squeeze()).reshape(
            (-1, 1)
        )
        r_R_R_grasp = self.robot.convert_pos_to_robot_frame(r_O_O_grasp)

        # self.robot._world.draw_cross(np.squeeze(r_O_O_grasp))

        # Compute desired orientation
        C_obj_grasp = R.from_quat(obj_info.grasp_orient[link_id][grasp_id])
        C_rob_ee_default = R.from_quat(self.robot.start_orient)
        C_rob_grasp = C_O_rob.inv() * C_O_obj * C_obj_grasp
        C_rob_ee = (
            C_rob_grasp * C_rob_ee_default
        )  # Apply standard EE orientation. EE will be in default orientation if robot and grasp orientation are equal

        return r_R_R_grasp[:3], C_rob_ee.as_quat()

    def grasp_object(self, target_name, link_idx=0, grasp_id=0, lock=None):
        if lock is not None:
            lock.acquire()
        pos, orient = self.compute_grasp(target_name, link_idx, grasp_id)

        self.robot.open_gripper()

        # Go to pre-grasp pose
        pos_pre = pos - R.from_quat(orient).apply(
            np.array([0.0, 0.0, self._pregrasp_z_offset])
        )
        pos_pre_joints = self.robot.ik(pos_pre, orient)
        if pos_pre_joints.tolist() is None:
            if lock is not None:
                lock.release()
            return False
        self.robot.transition_cmd_to(pos_pre_joints)
        self.robot._world.step_seconds(0.5)

        # Go to grasp pose
        try:
            self.robot.transition_cartesian(pos, orient)
        except IKError:
            return False

        self.robot._world.step_seconds(0.2)
        self.robot.close_gripper()
        self.robot._world.step_seconds(0.4)

        # Compute position of object link w.r.t. finger
        obj_info = self.scene.objects[target_name]
        target_uid = obj_info.model.uid
        target_link_id = obj_info.grasp_links[link_idx]
        r_O_O_finger, C_O_finger = self.robot.get_link_pose("panda_leftfinger")
        # r_O_O_finger = r_O_O_finger.reshape((-1, 1))
        C_O_finger = R.from_quat(C_O_finger)
        r_O_O_obj, C_O_obj = get_object_link_pose(target_uid, target_link_id)
        r_O_O_obj = np.reshape(r_O_O_obj, (3,))
        r_finger_finger_obj = C_O_finger.inv().apply(
            np.reshape(r_O_O_obj - r_O_O_finger, (3,))
        )
        C_finger_obj = C_O_finger.inv() * C_O_obj

        # Create no slip constraint between object and fingers
        constraint_spec = ConstraintSpec(
            self.robot.model.uid,
            self.robot.link_name_to_index["panda_leftfinger"],
            target_uid,
            target_link_id,
            r_finger_finger_obj,
            C_finger_obj.as_quat(),
        )
        self.robot._world.add_constraint(constraint_spec)

        # Save some variables required for releasing
        self.last_pre_pos = pos_pre
        self.last_pre_orient = orient
        self.robot.grasp_orientation = orient

        if lock is not None:
            lock.release()
        return True

    def release_object(self):
        pos_current, orient_current = self.robot.fk(np.array(self.robot.get_joints()))
        pos_retract = pos_current - np.matmul(
            R.from_quat(orient_current).as_matrix(), np.array([0.0, 0.0, 0.07])
        )

        self.robot.open_gripper()
        self.robot._world.step_seconds(0.5)
        self.robot.transition_cartesian(pos_retract, orient_current)


def get_grasping_description():
    action_name = "grasp"
    action_params = [["obj", "item-graspable"], ["gid", "grasp_id"], ["rob", "robot"]]
    action_preconditions = [
        ("in-reach", True, ["obj", "rob"]),
        ("empty-hand", True, ["rob"]),
        ("has-grasp", True, ["obj", "gid"]),
    ]
    action_effects = [
        ("empty-hand", False, ["rob"]),
        ("in-hand", True, ["obj", "rob"]),
        ("grasped-with", True, ["obj", "gid", "rob"]),
    ]
    action_exec_ignore_effects = list()
    return (
        action_name,
        {
            "params": action_params,
            "preconds": action_preconditions,
            "effects": action_effects,
            "exec_ignore_effects": action_exec_ignore_effects,
        },
    )
