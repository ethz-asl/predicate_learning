import os
import numpy as np
import pybullet as pb
import torch

from scipy.spatial.transform.rotation import Rotation

from highlevel_planning_py.predicate_learning.data_utils import shift_oabb
from highlevel_planning_py.sim.world import WorldPybullet
from highlevel_planning_py.tools_pl.util import ObjectInfo


class ObjectPerceptionInfo:
    def __init__(self, link_name: str, parent_spec):
        self.children = set()
        self.name = link_name
        self.parent = parent_spec

        # Object-aligned bounding box
        self.oabb = None

        # Vector from object frame to centroid
        self.centroid = None

    def __str__(self):
        info = (
            f"Object link name: {self.name}\n"
            f"Children: {self.children}\n"
            f"Parent: {self.parent}"
        )
        return info


class FakePerceptionPipeline:
    def __init__(self, logger, pb_client_id, paths):
        self.logger = logger
        self.pb_client_id = pb_client_id
        self.paths = paths

        self.perceived_objects = set()
        self.blacklist_objects = set()  # list of objects not to perceive (e.g. robot)
        self.object_info = {
            "objects_by_id": dict(),
            "object_ids_by_name": dict(),
            "sub_objects": dict(),
        }
        self.link_idx_to_name = dict()
        self.link_name_to_idx = dict()

    def reset(self):
        self.perceived_objects.clear()
        self.blacklist_objects.clear()  # list of objects not to perceive (e.g. robot)
        self.object_info["objects_by_id"].clear()
        self.object_info["object_ids_by_name"].clear()
        self.object_info["sub_objects"].clear()
        self.link_idx_to_name.clear()
        self.link_name_to_idx.clear()

    def callback_segmentation(self, segmentation_image):
        unique_values = np.unique(segmentation_image)

        # The segmentation combines the object unique id and link index as follows:
        # value = objectUniqueId + (linkIndex+1)<<24.
        # https://github.com/bulletphysics/bullet3/blob/master/examples/pybullet/examples/segmask_linkindex.py

        for unique_val in unique_values:
            if unique_val >= 0:
                object_idx = unique_val & ((1 << 24) - 1)
                link_idx = (unique_val >> 24) - 1
                object_name = None
                if object_idx not in self.blacklist_objects:
                    if (object_idx, link_idx) in self.object_info["objects_by_id"]:
                        object_name = self.object_info["objects_by_id"][
                            (object_idx, link_idx)
                        ].name
                    elif (object_idx, link_idx) in self.object_info["sub_objects"]:
                        parent = self.object_info["sub_objects"][(object_idx, link_idx)]
                        object_name = self.object_info["objects_by_id"][parent].name

                    if object_name:
                        self.perceived_objects.add(object_name)
                    else:
                        self.logger.warning(
                            f"[fpp]: skipping object with (bodyId, linkId)={object_idx, link_idx}."
                            f" Are all objects registered?",
                            throttle_duration_sec=1.0,
                        )

    def add_object_to_blacklist(self, object_id):
        self.blacklist_objects.add(object_id)

    def get_objects(self, observed_only: bool):
        ret = list()
        for name in self.object_info["object_ids_by_name"]:
            if (not observed_only) or (name in self.perceived_objects):
                ret.append(name)
        return ret

    def get_object_info(self, name: str, observed_only: bool):
        if observed_only and name not in self.perceived_objects:
            return None
        else:
            base_body_id, base_link_id = self.object_info["object_ids_by_name"][name]
            obj_info = self.object_info["objects_by_id"][(base_body_id, base_link_id)]

            # Bounding box
            aabb = self.compute_bounding_box(
                base_body_id, base_link_id, self.pb_client_id
            )

            # Orientation
            if base_link_id == -1:
                ret = pb.getBasePositionAndOrientation(
                    base_body_id, physicsClientId=self.pb_client_id
                )
            else:
                ret = pb.getLinkState(
                    base_body_id, base_link_id, physicsClientId=self.pb_client_id
                )
            urdf_pos = np.array(ret[0])
            orientation = np.array(ret[1])
            orient_r = Rotation.from_quat(orientation)

            # Centroid position
            com = urdf_pos + orient_r.apply(obj_info.centroid, inverse=False)

            # Size
            oabb = obj_info.oabb

            # Shifted OABB
            oabb_tensor = torch.unsqueeze(torch.tensor(oabb), 0).view(1, -1, 6)
            com_tensor = torch.unsqueeze(torch.tensor(com), 0).unsqueeze(1)
            orientation_tensor = torch.unsqueeze(
                torch.tensor(orientation), 0
            ).unsqueeze(1)
            shifted_oabb = shift_oabb(
                oabb_tensor, com_tensor, orientation_tensor, "cpu"
            )
            shifted_oabb = torch.squeeze(shifted_oabb).numpy()

            # Select OABB corners
            shifted_oabb_4corners = shifted_oabb[[0, 1, 2, 4], :]

            # OABB surface centers
            oabb_surface_corners = np.stack(
                (
                    shifted_oabb[4:, :],
                    shifted_oabb[[0, 2, 4, 6], :],
                    shifted_oabb[[0, 1, 4, 5], :],
                )
            )
            oabb_surface_centers = np.mean(oabb_surface_corners, axis=1)

            return (
                com,
                aabb,
                orientation,
                oabb,
                shifted_oabb_4corners,
                oabb_surface_centers,
            )

    def compute_bounding_box(
        self, base_body_id, base_link_id, pb_client_id, replacement_pb_uid=None
    ):
        pb_uid = base_body_id if replacement_pb_uid is None else replacement_pb_uid
        aabb = np.array(pb.getAABB(pb_uid, base_link_id, physicsClientId=pb_client_id))
        for child in self.object_info["objects_by_id"][
            (base_body_id, base_link_id)
        ].children:
            tmp = np.array(pb.getAABB(pb_uid, child[1], physicsClientId=pb_client_id))
            aabb[0, :] = np.minimum(aabb[0, :], tmp[0, :])
            aabb[1, :] = np.maximum(aabb[1, :], tmp[1, :])
        return aabb

    def register_object(self, uid, object_info: ObjectInfo, base_name: str = None):
        assert uid not in self.link_idx_to_name
        created_objects = list()
        self.link_idx_to_name[uid] = dict()
        num_joints = pb.getNumJoints(uid, physicsClientId=self.pb_client_id)
        tmp = pb.getBodyInfo(uid, physicsClientId=self.pb_client_id)
        b_name = tmp[1].decode("utf-8") if base_name is None else base_name
        self.link_idx_to_name[uid][-1] = b_name
        self.link_name_to_idx[b_name] = (uid, -1)
        self._create_object(uid, -1, b_name)
        created_objects.append((uid, -1))
        for i in range(num_joints):
            info = pb.getJointInfo(uid, i, physicsClientId=self.pb_client_id)
            link_name = info[12].decode("utf-8")
            self.link_idx_to_name[uid][i] = link_name
            self.link_name_to_idx[link_name] = (uid, i)
            parent_idx = info[16]

            if info[2] == pb.JOINT_FIXED:
                # This link is part of an already existing object
                if (uid, parent_idx) in self.object_info["objects_by_id"]:
                    self.object_info["objects_by_id"][(uid, parent_idx)].children.add(
                        (uid, i)
                    )
                    self.object_info["sub_objects"][(uid, i)] = (uid, parent_idx)
                else:
                    for potential_ancestor in self.object_info["objects_by_id"]:
                        if potential_ancestor[0] != uid:
                            continue
                        if (uid, parent_idx) in self.object_info["objects_by_id"][
                            potential_ancestor
                        ].children:
                            self.object_info["objects_by_id"][
                                potential_ancestor
                            ].children.add((uid, i))
                            self.object_info["sub_objects"][
                                (uid, i)
                            ] = potential_ancestor
                            break
                if (uid, i) not in self.object_info["sub_objects"]:
                    raise RuntimeError("Couldn't find parent")
            else:
                # A new object is started with this link
                self._create_object(
                    uid, i, f"{base_name}_{link_name}", (uid, parent_idx)
                )
                created_objects.append((uid, i))

        # Compute oabb for each created object
        tmp_world = WorldPybullet(style="direct", include_floor=False)
        obj_path = os.path.join(
            self.paths[object_info.urdf_relative_to], object_info.urdf_path
        )
        tmp_model = tmp_world.add_model(
            path=obj_path,
            position=np.array([0.0, 0.0, 0.0]),
            orientation=np.array([0.0, 0.0, 0.0, 1.0]),
            scale=object_info.scale,
            merge_fixed_links=object_info.merge_fixed_links,
        )
        for created_object in created_objects:
            oabb = self.compute_bounding_box(
                uid,
                created_object[1],
                tmp_world.client_id,
                replacement_pb_uid=tmp_model.uid,
            )
            com = np.mean(oabb, 0)
            oabb -= com
            self.object_info["objects_by_id"][created_object].oabb = oabb
            self.object_info["objects_by_id"][created_object].centroid = com
        tmp_world.close()

    def print_registered_objects(self):
        info_string = "[Fake Perception Pipeline] Registered objects list:"
        for object_ids, object_info in self.object_info["objects_by_id"].items():
            info_string += f"\n- (body id, link id)={object_ids},\n{object_info}"
        self.logger.info(info_string)

    def _create_object(self, body_id, link_id, name, parent_spec=None):
        assert name not in self.object_info["object_ids_by_name"]
        object_ids = (body_id, link_id)
        object_info = ObjectPerceptionInfo(name, parent_spec)
        self.object_info["objects_by_id"][object_ids] = object_info
        self.object_info["object_ids_by_name"][name] = object_ids

    def populate_from_scene(self, scene):
        for obj_name in scene.objects:
            self.register_object(
                uid=scene.objects[obj_name].model.uid,
                object_info=scene.objects[obj_name],
                base_name=obj_name,
            )

    def get_object_centroid(self, name):
        base_body_id, base_link_id = self.object_info["object_ids_by_name"][name]
        obj_info = self.object_info["objects_by_id"][(base_body_id, base_link_id)]
        return obj_info.centroid

    def get_object_oabb(self, name):
        base_body_id, base_link_id = self.object_info["object_ids_by_name"][name]
        obj_info = self.object_info["objects_by_id"][(base_body_id, base_link_id)]
        return obj_info.oabb
