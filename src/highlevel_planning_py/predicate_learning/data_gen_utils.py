import os
from typing import Tuple, List
import numpy as np
from scipy.spatial.transform import Rotation
import pybullet as p
import igibson
from igibson.objects.ycb_object import YCBObject
from igibson.objects.articulated_object import ArticulatedObject
from igibson import object_states
from highlevel_planning_py.tools_pl.util import ObjectInfo


# ----------- ON predicate ---------------------------------


class YCBObjectWrapper(YCBObject):
    def __init__(self, name, scale=1):
        super(YCBObjectWrapper, self).__init__(name, scale)

    def force_wakeup(self):
        pass

    def _load(self):
        collision_id = p.createCollisionShape(
            p.GEOM_MESH, fileName=self.collision_filename, meshScale=(self.scale,) * 3
        )
        visual_id = p.createVisualShape(
            p.GEOM_MESH, fileName=self.visual_filename, meshScale=(self.scale,) * 3
        )

        body_id = p.createMultiBody(
            baseCollisionShapeIndex=collision_id,
            baseVisualShapeIndex=visual_id,
            basePosition=[0.2, 0.2, 1.5],
            baseMass=0.1,
        )
        self.body_id = body_id
        return body_id


# Created to load objects without allowing sleeping
class ArticulatedObjectWrapper(ArticulatedObject):
    def _load(self):
        """
        Load the object into pybullet
        """
        flags = p.URDF_USE_MATERIAL_COLORS_FROM_MTL
        if self.merge_fixed_links:
            flags |= p.URDF_MERGE_FIXED_LINKS

        body_id = p.loadURDF(self.filename, globalScaling=self.scale, flags=flags)

        self.mass = p.getDynamicsInfo(body_id, -1)[0]
        self.body_id = body_id
        self.create_link_name_to_vm_map(body_id)
        return body_id


class FakeWorld:
    def __init__(self, cid):
        self.client_id = cid


def create_objects(
    num_objects: int,
    available_objects: List[str],
    name_prefix: str,
    scale_range: Tuple[float, float],
    paths: dict,
):
    """
    Samples from a list of available objects and creates igibson handles and object infos.

    Args:
        num_objects: Number of objects to sample
        available_objects: Items can be paths of objects anywhere or a string
            identifying an object in pybullet_data
        name_prefix: Prefix to use for naming objects
        scale_range: Range in which to sample scale for objects
        paths: dict with standard paths to assets, data, etc.

    Returns:
        object_handles: a dict with igibson object handles
        object_info: a dict with internal object info representation
    """
    object_handles = dict()
    object_info = dict()
    for i in range(num_objects):
        new_object_idx = np.random.randint(len(available_objects))
        new_object = available_objects[new_object_idx]
        new_scale = np.random.uniform(*scale_range)
        if "ycb" in new_object[1]:
            object_handles[f"{name_prefix}_{i}"] = YCBObjectWrapper(
                new_object[1].split("/")[-1], scale=new_scale
            )
            object_info[f"{name_prefix}_{i}"] = ObjectInfo(
                urdf_name_=new_object[1].split("/")[-1],
                urdf_path_=new_object[1],
                urdf_relative_to_=new_object[0],
                init_pos_=np.array([0, 0, 0]),
                init_orient_=np.array([0, 0, 0, 1]),
                init_scale_=new_scale,
            )
        elif "rbo" in new_object[1]:
            raise NotImplementedError("Not adapted to new relative path handling yet.")
            # object_handles[f"{name_prefix}_{i}"] = RBOObject(
            #     new_object[1].split("/")[-1], scale=new_scale
            # )
            # object_info[f"{name_prefix}_{i}"] = ObjectInfo(
            #     urdf_name_=new_object[1].split("/")[-1],
            #     urdf_path_=object_handles[f"{name_prefix}_{i}"].filename,
            #     init_pos_=np.array([0, 0, 0]),
            #     init_orient_=np.array([0, 0, 0, 1]),
            #     init_scale_=new_scale,
            # )
        else:
            new_object_path = os.path.join(paths[new_object[0]], new_object[1])
            object_handles[f"{name_prefix}_{i}"] = ArticulatedObjectWrapper(
                new_object_path, scale=new_scale, merge_fixed_links=False
            )
            object_info[f"{name_prefix}_{i}"] = ObjectInfo(
                urdf_path_=new_object[1],
                urdf_relative_to_=new_object[0],
                init_pos_=np.array([0, 0, 0]),
                init_orient_=np.array([0, 0, 0, 1]),
                init_scale_=new_scale,
                merge_fixed_links_=False,
            )
    return object_handles, object_info


def get_body_id(obj_handle):
    if hasattr(obj_handle, "get_body_id"):
        return obj_handle.get_body_id()
    else:  # Needed for igibson 2.1.0 and later
        ids = obj_handle.get_body_ids()
        assert len(ids) == 1
        return ids[0]


def check_in_collision_pair(uid1, uid2, client_id, tolerance=0.0):
    temp = p.getClosestPoints(uid1, uid2, distance=0.05, physicsClientId=client_id)
    for elem in temp:
        contact_distance = elem[8]
        if contact_distance < -tolerance:
            # There is a collision
            return True
    return False


def check_in_collision(object_dict, client_id, tolerance=0.0, one_to_all: str = None):
    object_list = list(object_dict.keys())
    if one_to_all is None:
        for obj1_idx in range(len(object_list) - 1):
            for obj2_idx in range(obj1_idx + 1, len(object_list)):
                uid1 = get_body_id(object_dict[object_list[obj1_idx]])
                uid2 = get_body_id(object_dict[object_list[obj2_idx]])
                res = check_in_collision_pair(uid1, uid2, client_id, tolerance)
                if res:
                    return True
    else:
        for obj2 in object_list:
            if obj2 == one_to_all:
                continue
            uid1 = get_body_id(object_dict[one_to_all])
            uid2 = get_body_id(object_dict[obj2])
            res = check_in_collision_pair(uid1, uid2, client_id, tolerance)
            if res:
                return True
    return False


def place_randomly(remaining_objects, ig_object_handles, drop_range_xy: float = 1.5):
    for other_obj in remaining_objects:
        new_pos = np.random.uniform(
            [-drop_range_xy, -drop_range_xy, 0.2], [drop_range_xy, drop_range_xy, 2.0]
        )
        new_orient = Rotation.random().as_quat()
        ig_object_handles[other_obj].set_position_orientation(new_pos, new_orient)


def place_on(supported, supporting, ig_object_handles, ig_fixed_object_handles):
    res_on = (
        ig_object_handles[supported]
        .states[object_states.OnTop]
        .set_value(
            ig_fixed_object_handles[supporting], True, use_ray_casting_method=True
        )
    )
    return res_on


def find_ycb_objects():
    ycb_dir = os.path.join(igibson.assets_path, "models", "ycb")
    available_objects_ = os.listdir(ycb_dir)
    available_objects_.sort()
    ycb_objects = list()
    for obj in available_objects_:
        path = os.path.join(ycb_dir, obj)
        if os.path.isdir(path):
            obj = ("igibson_dir", os.path.join("models", "ycb", obj))
            ycb_objects.append(obj)
    assert len(ycb_objects) == 21
    return ycb_objects


# ----------- INSIDE predicate ---------------------------------


def create_objects_pybullet(
    num_objects: int,
    available_objects: List[str],
    name_prefix: str,
    scale_range: Tuple[float, float],
    paths: dict,
    world,
):
    object_handles = dict()
    object_info = dict()
    for i in range(num_objects):
        new_object_idx = np.random.randint(len(available_objects))
        new_object = available_objects[new_object_idx]
        new_scale = np.random.uniform(*scale_range)
        new_object_path = os.path.join(paths[new_object[0]], new_object[1])
        object_handles[f"{name_prefix}_{i}"] = world.add_model(
            new_object_path,
            position=[0, 0, 2.0],
            orientation=[0, 0, 0, 1],
            scale=new_scale,
        )
        object_info[f"{name_prefix}_{i}"] = ObjectInfo(
            urdf_path_=new_object[1],
            urdf_relative_to_=new_object[0],
            init_pos_=np.array([0, 0, 2.0]),
            init_orient_=np.array([0, 0, 0, 1]),
            init_scale_=new_scale,
            merge_fixed_links_=False,
        )
    return object_handles, object_info


def place_inside(
    contained, container, ig_object_handles, ig_fixed_object_handles, perception
):
    # Info about objects
    container_centroid = perception.get_object_centroid(container)
    contained_centroid = perception.get_object_centroid(contained)
    container_oabb = perception.get_object_oabb(container)
    contained_oabb = perception.get_object_oabb(contained)

    # Max extent of contained
    contained_min_extent = np.min(contained_oabb[1, :])

    # Sample orientation
    upright = bool(np.random.randint(2))
    if upright:
        new_orient = Rotation.identity()
    else:
        new_orient = Rotation.random()

    # Sample position
    sample_space = [
        container_oabb[0, :] + contained_min_extent,
        container_oabb[1, :] - contained_min_extent,
    ]
    new_com_pos = np.random.uniform(sample_space[0], sample_space[1])
    new_base_pos = new_com_pos - new_orient.apply(contained_centroid)

    # Move to container
    container_pos, container_orient = ig_fixed_object_handles[
        container
    ].get_position_orientation()
    new_base_pos += container_pos
    com_adjustment = Rotation.from_quat(container_orient).apply(container_centroid)
    new_base_pos += com_adjustment

    ig_object_handles[contained].set_position_orientation(
        new_base_pos, new_orient.as_quat()
    )

    # Check for collisions with container
    while check_in_collision_pair(
        ig_object_handles[contained].get_body_id(),
        ig_fixed_object_handles[container].get_body_id(),
        ig_object_handles[contained]._physics_client,
        tolerance=0.005,
    ):
        new_base_pos += np.array([0, 0, 0.02])
        ig_object_handles[contained].set_position_orientation(
            new_base_pos, new_orient.as_quat()
        )

    # Avoid collisions with other objects
    while check_in_collision(
        ig_object_handles,
        ig_object_handles[contained]._physics_client,
        tolerance=0.005,
        one_to_all=contained,
    ):
        new_base_pos += np.array([0, 0, 0.02])
        ig_object_handles[contained].set_position_orientation(
            new_base_pos, new_orient.as_quat()
        )
