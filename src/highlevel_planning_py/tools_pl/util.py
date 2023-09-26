# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#  Predicate learning framework
#  Copyright (C) 2023. ETH ASL
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <https://www.gnu.org/licenses/>.
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import os
import ast
from scipy.spatial.transform import Rotation as R
import numpy as np
import pybullet as pb
import matplotlib.pyplot as plt
import argparse
import tracemalloc
import linecache


def dict_set_without_overwrite(dict_in, key, value):
    if key in dict_in:
        assert dict_in[key] == value
    else:
        dict_in[key] = value
    return dict_in


def quat_from_rpy(orient_rpy):
    orient_rep = R.from_euler("xyz", orient_rpy.tolist())
    return orient_rep.as_quat()


def quat_from_rpy_yaw_first(orient_rpy):
    orient_rep = R.from_euler("zxy", orient_rpy.tolist())
    return orient_rep.as_quat()


def quat_from_rotvec(rotvec):
    orient_rep = R.from_rotvec(rotvec)
    return orient_rep.as_quat()


def quat_from_mat(mat):
    orient_rep = R.from_matrix(mat)
    return orient_rep.as_quat()


def rotate_orient(orig, axis="z", deg=0.0):
    r = R.from_quat(orig)
    assert axis == "x" or axis == "y" or axis == "z", "Invalid axis"
    op = R.from_euler(axis, deg, degrees=True)
    res = op * r
    return res.as_quat()


def homogenous_trafo(translation, rotation):
    assert type(translation) is np.ndarray
    assert len(translation) == 3
    assert type(rotation) is R
    t = np.concatenate((rotation.as_matrix(), translation.reshape(-1, 1)), axis=1)
    t = np.concatenate((t, np.array([[0.0, 0.0, 0.0, 1.0]])), axis=0)
    return t


def pos_and_orient_from_hom_trafo(hom_trafo):
    return hom_trafo[:3, 3], R.from_matrix(hom_trafo[:3, :3]).as_quat()


def invert_hom_trafo(hom_trafo):
    res = np.eye(4)
    res[:3, :3] = np.transpose(hom_trafo[:3, :3])
    res[:3, 3] = -np.matmul(np.transpose(hom_trafo[:3, :3]), hom_trafo[:3, 3])
    return res


def get_combined_aabb(uid):
    """
    By default, the getAABB function offered by pybullet only outputs the bounding
    box for a single link (the base link by default). This function combines the 
    AABBs of all links an object has.
    
    Args:
        uid (int): The UID of the object we would like to compute the bounding box of.
    """
    num_joints = pb.getNumJoints(uid)
    aabb_min, aabb_max = pb.getAABB(uid, linkIndex=-1)
    aabb_min, aabb_max = np.array(aabb_min), np.array(aabb_max)
    for i in range(num_joints):
        aabb_local_min, aabb_local_max = pb.getAABB(uid, linkIndex=i)
        aabb_local_min, aabb_local_max = (
            np.array(aabb_local_min),
            np.array(aabb_local_max),
        )
        aabb_min = np.minimum(aabb_min, aabb_local_min)
        aabb_max = np.maximum(aabb_max, aabb_local_max)
    return aabb_min, aabb_max


def get_object_position(object_name, scene_objects, knowledge_base):
    if object_name in scene_objects:
        pos, _ = pb.getBasePositionAndOrientation(scene_objects[object_name].model.uid)
        return np.array(pos)
    elif object_name in knowledge_base.lookup_table:
        return knowledge_base.lookup_table[object_name]
    else:
        raise ValueError("Invalid object")


def capture_image_pybullet(
    client_id, path=None, show=False, camera_pos=(2, 2, 1), target_pos=(0, 0, 0)
):
    width = 640
    height = 480
    fov = 55
    near = 0.1
    far = 20.0
    aspect = width / height
    up_vector = [0, 0, 1]
    view_matrix = pb.computeViewMatrix(camera_pos, target_pos, up_vector)
    projection_matrix = pb.computeProjectionMatrixFOV(fov, aspect, near, far)

    res = pb.getCameraImage(
        width,
        height,
        view_matrix,
        projection_matrix,
        renderer=pb.ER_TINY_RENDERER,
        flags=pb.ER_NO_SEGMENTATION_MASK,
        physicsClientId=client_id,
    )
    rgb_pixels = res[2]

    if type(rgb_pixels) is tuple:
        # For some reason, on the cluster, rgb_pixels is returned as a tuple, so need to reformat.
        # This is despite using same pybullet version on cluster and my machine.
        rgb_pixels = np.array(rgb_pixels)
        rgb_pixels = np.reshape(rgb_pixels, (height, width, -1))

    if show or path is not None:
        plt.imshow(rgb_pixels)
        plt.axis("off")
    if show:
        plt.show()
    if path is not None:
        plt.savefig(path, bbox_inches="tight")
    return rgb_pixels


def dir_levels_up(filepath, num_levels_up):
    res = filepath
    for _ in range(num_levels_up):
        res = os.path.dirname(res)
    return res


class IKError(Exception):
    pass


class SkillExecutionError(Exception):
    pass


class ObjectInfo:
    def __init__(
        self,
        urdf_path_,
        init_pos_,
        init_orient_,
        init_scale_=1.0,
        grasp_links_=None,
        grasp_pos_=None,
        grasp_orient_=None,
        model_=None,
        nav_angle_=None,
        nav_min_dist_=None,
        friction_setting_=None,
        joint_setting_=None,
        urdf_name_=None,
        merge_fixed_links_=False,
        force_fixed_base_=False,
        urdf_relative_to_="",
    ):
        if grasp_links_ is None:
            grasp_links_ = list()
        if grasp_pos_ is None:
            grasp_pos_ = dict()
        if grasp_orient_ is None:
            grasp_orient_ = dict()
        assert type(grasp_pos_) is dict
        assert type(grasp_orient_) is dict
        assert type(grasp_links_) is list
        self.urdf_path = urdf_path_
        self.urdf_relative_to = urdf_relative_to_
        self.urdf_name = urdf_name_ if urdf_name_ is not None else urdf_path_
        self.init_pos = init_pos_
        self.init_orient = init_orient_
        self.scale = init_scale_
        self.grasp_pos = grasp_pos_
        self.grasp_orient = grasp_orient_
        self.model = model_
        self.nav_angle = nav_angle_
        self.nav_min_dist = nav_min_dist_
        self.grasp_links = grasp_links_
        self.friction_setting = friction_setting_
        self.joint_setting = joint_setting_
        self.merge_fixed_links = merge_fixed_links_
        self.force_fixed_base = force_fixed_base_

    def __eq__(self, other):
        res = True
        res &= self.urdf_name == other.urdf_name
        # res &= np.array_equal(self.init_pos, other.init_pos)
        # res &= self.init_orient == other.init_orient
        res &= self.scale == other.scale
        # res &= self.grasp_pos == other.grasp_pos
        # res &= self.grasp_orient == other.grasp_orient
        # res &= self.nav_angle == other.nav_angle
        # res &= self.nav_min_dist == other.nav_min_dist
        # res &= self.grasp_links == other.grasp_links
        # res &= self.friction_setting == other.friction_setting
        # res &= self.joint_setting == other.joint_setting
        res &= self.merge_fixed_links == other.merge_fixed_links
        res &= self.force_fixed_base == other.force_fixed_base
        return res


class ConstraintSpec:
    def __init__(
        self,
        parent_uid,
        parent_link_id,
        child_uid,
        child_link_id,
        trafo_pos,
        trafo_orient,
    ):
        self.parent_uid = parent_uid
        self.parent_link_id = parent_link_id
        self.child_uid = child_uid
        self.child_link_id = child_link_id
        self.trafo_pos = trafo_pos
        self.trafo_orient = trafo_orient

        self.constrain_id = None


def parse_arguments(args_to_parse=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-r",
        "--reuse-objects",
        action="store_true",
        help="if given, the simulation does not reload objects. Objects must already be present.",
    )
    parser.add_argument(
        "-s",
        "--sleep",
        action="store_true",
        help="if given, the simulation will sleep for each update step, to mimic real time execution.",
    )
    parser.add_argument(
        "-m",
        "--method",
        action="store",
        choices=["gui", "direct", "shared"],
        type=str,
        help="determines in which mode to connect to pybullet.",
        default="gui",
    )
    parser.add_argument(
        "-n", "--noninteractive", action="store_true", help="skip user prompts."
    )
    parser.add_argument(
        "--no-seed",
        action="store_true",
        help="if given, RNGs are not initialized with a random seed.",
    )
    parser.add_argument(
        "-d",
        "--domain-file",
        action="store",
        default="_domain.pkl",
        help="The file name of the domain file loaded/stored by the knowledge base.",
    )
    parser.add_argument(
        "-c",
        "--config-file-path",
        action="store",
        help="Absolute path to the config file to use.",
        default="",
    )
    if args_to_parse is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(args=args_to_parse)
    return args


def string_to_list(string: str):
    if string.lower() == "none" or string == "0":
        return None
    return ast.literal_eval(string)


def string_to_bool(string: str) -> bool:
    string_lower = string.lower()
    if string_lower == "true" or string_lower == "1":
        return True
    elif string_lower == "false" or string_lower == "0":
        return False
    else:
        raise ValueError


def check_path_exists(path_to_check):
    if not os.path.isdir(path_to_check):
        os.makedirs(path_to_check)


def exit_handler(rep):
    rep.write_result_file()


def parse_arguments_pddl_bm(args_to_parse=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-n", "--noninteractive", action="store_true", help="skip user prompts."
    )
    parser.add_argument(
        "--no-seed",
        action="store_true",
        help="if given, RNGs are not initialized with a random seed.",
    )
    parser.add_argument(
        "-d",
        "--domain-file",
        action="store",
        default="_domain.pkl",
        help="The file name of the domain file loaded/stored by the knowledge base.",
    )
    parser.add_argument(
        "-c",
        "--config-file-path",
        action="store",
        help="Absolute path to the config file to use.",
        default="",
    )
    if args_to_parse is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(args=args_to_parse)
    return args


def memory_display_top(snapshot, key_type="lineno", limit=6):
    snapshot = snapshot.filter_traces(
        (
            tracemalloc.Filter(False, "<frozen importlib._bootstrap>"),
            tracemalloc.Filter(False, "<unknown>"),
        )
    )
    top_stats = snapshot.statistics(key_type)

    print("Top %s lines" % limit)
    for index, stat in enumerate(top_stats[:limit], 1):
        frame = stat.traceback[0]
        # replace "/path/to/module/file.py" with "module/file.py"
        filename = os.sep.join(frame.filename.split(os.sep)[-2:])
        print(
            "#%s: %s:%s: %.1f KiB" % (index, filename, frame.lineno, stat.size / 1024)
        )
        line = linecache.getline(frame.filename, frame.lineno).strip()
        if line:
            print("    %s" % line)

    other = top_stats[limit:]
    if other:
        size = sum(stat.size for stat in other)
        print("%s other: %.1f KiB" % (len(other), size / 1024))
    total = sum(stat.size for stat in top_stats)
    print("Total allocated size: %.1f KiB" % (total / 1024))


def get_weights_list(training_dir, model_label):
    available_weights = os.listdir(os.path.join(training_dir, "models"))
    available_weights = [
        item
        for item in available_weights
        if f"{model_label}_model" in item and "final" not in item
    ]
    available_weights.sort()
    return available_weights


def get_latest_weights(training_dir, model_label):
    available_weights = get_weights_list(training_dir, model_label)
    weights_file = available_weights[-1]
    weights_idx = int(".".join(weights_file.split(".")[:-1]).split("_")[-1])
    return weights_file, weights_idx
