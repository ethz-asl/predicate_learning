from highlevel_planning_py.skills.grasping import SkillGrasping
from highlevel_planning_py.tools_pl.util import get_combined_aabb, SkillExecutionError
import pybullet as p
import numpy as np
from scipy.spatial.transform import Rotation


class PredicatesBase:
    def __init__(self, client_id):
        self.pb_client_id = client_id

    def on_(self, supporting_uid, supported_uid, above_tol):
        aabb_supporting = get_combined_aabb(supporting_uid)

        pos_supported, _ = p.getBasePositionAndOrientation(
            supported_uid, physicsClientId=self.pb_client_id
        )
        pos_supported = np.array(pos_supported)
        aabb_supported = get_combined_aabb(supported_uid)

        lower_supporting = aabb_supporting[0]
        upper_supporting = aabb_supporting[1]
        lower_supported = aabb_supported[0]

        # Check if supported object is above supporting one (z-coordinate)
        above = (
            upper_supporting[2] - above_tol
            < lower_supported[2]
            < upper_supporting[2] + above_tol
        )

        # Check if supported object is within footprint of supporting one (xy-plane).
        # Currently, this is based on the position of the supported object. Need to see whether this makes sense.
        within = np.all(
            np.greater_equal(pos_supported[:2], lower_supporting[:2])
        ) and np.all(np.less_equal(pos_supported[:2], upper_supporting[:2]))

        return above and within

    def inside_oabb(
        self,
        container_uid,
        contained_uid,
        container_oabb,
        contained_oabb,
        container_centroid,
        contained_centroid,
    ):
        # Get all corners of contained OABB
        corners = np.array(
            [
                np.array([x, y, z])
                for x in [contained_oabb[0][0], contained_oabb[1][0]]
                for y in [contained_oabb[0][1], contained_oabb[1][1]]
                for z in [contained_oabb[0][2], contained_oabb[1][2]]
            ]
        )

        container_pos, container_orient = p.getBasePositionAndOrientation(
            container_uid, physicsClientId=self.pb_client_id
        )
        container_pos, container_orient = (
            np.array(container_pos),
            np.array(container_orient),
        )
        contained_pos, contained_orient = p.getBasePositionAndOrientation(
            contained_uid, physicsClientId=self.pb_client_id
        )
        contained_pos, contained_orient = (
            np.array(contained_pos),
            np.array(contained_orient),
        )

        # Transform corners to container frame
        corners += contained_centroid
        container_rot = Rotation.from_quat(container_orient)
        contained_rot = Rotation.from_quat(contained_orient)
        corners = contained_rot.apply(corners)
        corners += contained_pos - container_pos
        corners = container_rot.inv().apply(corners)
        corners = corners - container_centroid

        # Check if all corners are within container OABB
        res = np.all(np.greater_equal(corners, container_oabb[0])) and np.all(
            np.less_equal(corners, container_oabb[1])
        )
        return res


class Predicates(PredicatesBase):
    def __init__(self, scene, robot, knowledge_base, cfg, pb_client_id):
        super(Predicates, self).__init__(pb_client_id)
        self.call = {
            "empty-hand": self.empty_hand,
            "in-hand": self.in_hand,
            "in-reach": self.in_reach,
            "at": self.at,
            "inside": self.inside,
            "on": self.on,
            "has-grasp": self.has_grasp,
            "grasped-with": self.grasped_with,
        }

        self.descriptions = {
            "empty-hand": [["rob", "robot"]],
            "in-hand": [["obj", "item"], ["rob", "robot"]],
            "in-reach": [["target", "navgoal"], ["rob", "robot"]],
            "at": [["target", "navgoal"], ["rob", "robot"]],
            "inside": [["container", "item"], ["contained", "item"]],
            "on": [["supporting", "item"], ["supported", "item"]],
            "has-grasp": [["obj", "navgoal"], ["gid", "grasp_id"]],
            "grasped-with": [["obj", "item"], ["gid", "grasp_id"], ["rob", "robot"]],
        }

        self.sk_grasping = SkillGrasping(scene, robot, cfg)
        self._scene = scene
        self._robot_uid = robot.model.uid
        self._robot = robot
        self._kb = knowledge_base
        self._cfg = cfg

    def empty_hand(self, robot_name):
        robot = self._robot
        grasped_sth = robot.check_grasp()
        return not grasped_sth

    def in_hand(self, target_object, robot_name):
        robot = self._robot
        empty_hand_res = self.empty_hand(robot_name)
        temp = p.getClosestPoints(
            self._robot_uid,
            self._scene.objects[target_object].model.uid,
            distance=0.01,
            physicsClientId=self.pb_client_id,
        )
        dist_finger1 = 100
        dist_finger2 = 100
        for contact in temp:
            if contact[3] == robot.joint_idx_fingers[0]:
                dist_finger1 = contact[8] if contact[8] < dist_finger1 else dist_finger1
            elif contact[3] == robot.joint_idx_fingers[1]:
                dist_finger2 = contact[8] if contact[8] < dist_finger2 else dist_finger2
        desired_object_in_hand = (abs(dist_finger1) < 0.01) and (
            abs(dist_finger2) < 0.01
        )
        return (not empty_hand_res) and desired_object_in_hand

    def in_reach(self, target_item, robot_name):
        if self._kb.is_type(target_item, "position"):
            return self.in_reach_pos(self._kb.lookup_table[target_item], robot_name)
        elif type(target_item) is str:
            return self.in_reach_obj(target_item, robot_name)
        else:
            raise ValueError

    def in_reach_obj(self, target_object, robot_name):
        """
        Check if an object in the scene is in reach of the robot arm.
        
        Arguments:
            target_object (string): The object name, as stated in the scene
                                      file and/or the problem PDDL.
            robot_name (RobotArm): The interface class of the robot arm to use.
        
        Returns:
            bool: Whether the object can be grasped from the robot's current position.
        """
        try:
            pos, orient = self.sk_grasping.compute_grasp(target_object, 0, 0)
        except SkillExecutionError:
            return False
        cmd = self._robot.ik(pos, orient)
        if cmd.tolist() is None or cmd is None:
            return False
        else:
            return True

    def in_reach_pos(self, target_pos, robot_name):
        """
        Similar to function "in_reach", but takes position instead of object name.
        
        Args:
            target_pos (list): 3D vector of position to query
            robot_name ([type]): [description]
        
        Returns:
            [type]: [description]
        """
        # Convert position into robot base frame
        r_O_O_pos = target_pos
        r_R_R_grasp = self._robot.convert_pos_to_robot_frame(r_O_O_pos)

        cmd = self._robot.ik(r_R_R_grasp, self._robot.start_orient)
        if cmd.tolist() is None or cmd is None:
            return False
        else:
            return True

    def at(self, target_object, robot_name, use_closest_points=True):
        distance_limit = self._cfg.getparam(
            ["predicates", "at", "max_distance"], default_value=1.0
        )
        if self._kb.is_type(target_object, "position"):
            pos_object = self._kb.lookup_table[target_object]
            pos_robot, _ = self._robot.get_link_pose("ridgeback_dummy")
            distance = np.linalg.norm(pos_robot[:2] - pos_object[:2])
            return distance < distance_limit
        elif not use_closest_points:
            obj_info = self._scene.objects[target_object]
            target_id = obj_info.model.uid
            temp = p.getBasePositionAndOrientation(
                target_id, physicsClientId=self.pb_client_id
            )
            pos_object = temp[0]
            pos_robot, _ = self._robot.get_link_pose("ridgeback_dummy")
            distance = np.linalg.norm(pos_robot[:2] - pos_object[:2])
            return distance < distance_limit
        else:
            temp = p.getClosestPoints(
                self._robot_uid,
                self._scene.objects[target_object].model.uid,
                distance=1.1,
                physicsClientId=self.pb_client_id,
            )
            for point in temp:
                if point[8] < distance_limit:
                    return True
            return False

    def inside(self, container_object, contained_object):
        """
        Checks if one object in the scene is inside another.
        
        Args:
            container_object (string): Object name of the container object.
            contained_object (string): Object name of the contained object.
        
        Returns:
            bool: Whether the container object contains the contained one.
        """
        if contained_object == container_object:
            return False
        container_uid = self._scene.objects[container_object].model.uid
        contained_uid = self._scene.objects[contained_object].model.uid
        aabb_container = get_combined_aabb(container_uid)
        aabb_contained = get_combined_aabb(contained_uid)
        return np.all(np.less_equal(aabb_container[0], aabb_contained[0])) and np.all(
            np.greater_equal(aabb_container[1], aabb_contained[1])
        )

    def on(self, supporting_object, supported_object):
        """
        [summary]
        
        Args:
            supporting_object ([type]): [description]
            supported_object ([type]): [description]
        """
        if supporting_object == supported_object:
            return False
        supporting_uid = self._scene.objects[supporting_object].model.uid
        supported_uid = self._scene.objects[supported_object].model.uid
        above_tol = self._cfg.getparam(
            ["predicates", "on-pred", "max_above"], default_value=0.05
        )
        return super(Predicates, self).on_(supporting_uid, supported_uid, above_tol)

    def has_grasp(self, obj, gid):
        if obj in self._scene.objects:
            grasp_spec = self._kb.lookup_table[gid]
            success = True
            success &= 0 <= grasp_spec[0] < len(self._scene.objects[obj].grasp_links)
            if not success:
                return False
            link_id = self._scene.objects[obj].grasp_links[grasp_spec[0]]
            success &= (
                0 <= grasp_spec[1] < len(self._scene.objects[obj].grasp_pos[link_id])
            )
            return success
        else:
            return False

    def grasped_with(self, obj, gid, rob):
        success = True
        success &= self.has_grasp(obj, gid)
        success &= self.in_hand(obj, rob)
        return success
