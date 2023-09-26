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
from copy import deepcopy
from typing import Dict
import pybullet as pb


class SceneBase:
    def __init__(self, world, paths: Dict, restored_objects=None):
        self.world = world
        self.objects = dict()
        self.paths = paths

        self.initial_state = None

        self.user_debug_text_ids = set()

        if restored_objects is not None:
            self.objects = restored_objects

    def set_objects(self, objects):
        self.objects = deepcopy(objects)

    def add_objects(self, force_load=False):
        for key, obj in self.objects.items():
            if obj.model is None or force_load:
                mfl = (
                    obj.merge_fixed_links
                    if "merge_fixed_links" in obj.__dir__()
                    else False
                )  # This is for when we handle data that was created before ObjectInfo had this attribute.
                ffb = (
                    obj.force_fixed_base if hasattr(obj, "force_fixed_base") else False
                )  # This is for when we handle data that was created before ObjectInfo had this attribute.
                obj_path = os.path.join(self.paths[obj.urdf_relative_to], obj.urdf_path)
                self.objects[key].model = self.world.add_model(
                    obj_path,
                    obj.init_pos,
                    obj.init_orient,
                    scale=obj.scale,
                    merge_fixed_links=mfl,
                    force_fixed_base=ffb,
                )
            if self.objects[key].friction_setting is not None:
                for spec in self.objects[key].friction_setting:
                    pb.changeDynamics(
                        self.objects[key].model.uid,
                        self.objects[key].model.link_name_to_index[spec["link_name"]],
                        lateralFriction=spec["lateral_friction"],
                        physicsClientId=self.world.client_id,
                    )
            if self.objects[key].joint_setting is not None:
                for spec in self.objects[key].joint_setting:
                    pb.setJointMotorControl2(
                        self.objects[key].model.uid,
                        spec["jnt_idx"],
                        controlMode=spec["mode"],
                        force=spec["force"],
                        physicsClientId=self.world.client_id,
                    )
        self.initial_state = self.world.save_state()

    def restore_initial_poses(self):
        self.world.restore_state(self.initial_state)

    def show_object_labels(self):
        # Remove old ones
        for text_id in self.user_debug_text_ids:
            pb.removeUserDebugItem(text_id, physicsClientId=self.world.client_id)

        # Add new ones
        for obj in self.objects:
            tmp = pb.getDynamicsInfo(
                self.objects[obj].model.uid,
                linkIndex=-1,
                physicsClientId=self.world.client_id,
            )
            if tmp[0] < 0.01:
                link_idx = 0
            else:
                link_idx = -1

            new_id = pb.addUserDebugText(
                obj,
                [0, 0, 0],
                parentObjectUniqueId=self.objects[obj].model.uid,
                parentLinkIndex=link_idx,
                physicsClientId=self.world.client_id,
            )
            self.user_debug_text_ids.add(new_id)
