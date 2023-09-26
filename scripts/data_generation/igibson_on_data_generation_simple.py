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

import numpy as np
from igibson import object_states
import os
from tqdm import tqdm

import pybullet as p

from scipy.spatial.transform.rotation import Rotation

import igibson
from igibson.objects.articulated_object import ArticulatedObject
from igibson.objects.ycb_object import YCBObject
from igibson.simulator import Simulator
from igibson.scenes.empty_scene import EmptyScene

from highlevel_planning_py.tools_pl.util import ObjectInfo
from highlevel_planning_py.knowledge.predicates import PredicatesBase
from highlevel_planning_py.predicate_learning.demonstrations import (
    PredicateDemonstrationManager,
)
from highlevel_planning_py.sim.scene_base import SceneBase
from highlevel_planning_py.sim.fake_perception import FakePerceptionPipeline
from highlevel_planning_py.tools_pl.logger_stdout import LoggerStdout


PATHS = {
    "data_dir": os.path.join(os.path.expanduser("~"), "Data", "highlevel_planning")
}


class YCBObjectWrapper(YCBObject):
    def __init__(self, name, scale=1):
        super(YCBObjectWrapper, self).__init__(name, scale)

    def force_wakeup(self):
        pass


class FakeWorld:
    def __init__(self, cid):
        self.client_id = cid


def main():
    logger = LoggerStdout()
    scene_objects_fixed = dict()

    cabinet_lower_path = os.path.join(
        igibson.assets_path, "models/cabinet2/cabinet_0007.urdf"
    )
    scene_objects_fixed["cabinet_lower"] = ObjectInfo(
        urdf_name_="cabinet_0007.urdf",
        urdf_path_=cabinet_lower_path,
        init_pos_=np.array([-0.5, 0, 0.5]),
        init_orient_=np.array([0, 0, 0, 1]),
        merge_fixed_links_=True,
    )

    cabinet_upper_path = os.path.join(
        igibson.assets_path, "models/cabinet/cabinet_0004.urdf"
    )
    scene_objects_fixed["cabinet_upper"] = ObjectInfo(
        urdf_name_="cabinet_0004.urdf",
        urdf_path_=cabinet_upper_path,
        init_pos_=np.array([1.0, 0, 0.5]),
        init_orient_=np.array([0, 0, 0, 1]),
        merge_fixed_links_=True,
    )

    # Find available objects
    ycb_dir = os.path.join(igibson.assets_path, "models", "ycb")
    available_objects_ = os.listdir(ycb_dir)
    available_objects = list()
    for obj in available_objects_:
        if os.path.isdir(os.path.join(ycb_dir, obj)):
            available_objects.append(obj)
    del available_objects_

    # Iterate over scenes
    num_samples = 200
    num_samples_per_scene = 15  # each sample stands for a positive and a negative
    num_scenes = int(np.floor(num_samples / num_samples_per_scene))
    samples_per_scene = [num_samples_per_scene] * num_scenes
    samples_per_scene.append(num_samples % num_samples_per_scene)
    for scene_idx in tqdm(range(len(samples_per_scene))):
        # Select objects and scales
        num_objects = 6
        scene_objects = dict()
        ig_object_handles = dict()
        ig_fixed_object_handles = dict()
        for i in range(num_objects):
            new_object = np.random.choice(available_objects)
            new_scale = np.random.uniform(0.7, 1.3)
            scene_objects[f"obj{i}"] = ObjectInfo(
                urdf_name_=new_object,
                urdf_path_=os.path.join(ycb_dir, new_object),
                init_pos_=np.array([0, 0, 0]),
                init_orient_=np.array([0, 0, 0, 1]),
                init_scale_=new_scale,
            )
            ig_object_handles[f"obj{i}"] = YCBObjectWrapper(new_object, scale=new_scale)
        for obj_name in scene_objects_fixed:
            ig_fixed_object_handles[obj_name] = ArticulatedObject(
                filename=scene_objects_fixed[obj_name].urdf_path
            )

        # Set up simulator
        s = Simulator(mode="headless")
        scene = EmptyScene()
        predicates = PredicatesBase(s.cid)
        s.import_scene(scene)
        for obj_name, obj in ig_fixed_object_handles.items():
            s.import_object(obj)
            obj.set_position_orientation(
                scene_objects_fixed[obj_name].init_pos,
                scene_objects_fixed[obj_name].init_orient,
            )
        for obj_name, obj in ig_object_handles.items():
            s.import_object(obj)
        fake_world = FakeWorld(s.cid)
        fake_scene = SceneBase(fake_world, {})
        fake_scene.set_objects({**scene_objects_fixed, **scene_objects})
        fake_perception = FakePerceptionPipeline(logger, s.cid)
        for obj_name in fake_scene.objects:
            if obj_name in ig_object_handles:
                uid = ig_object_handles[obj_name].get_body_id()
                obj_info = scene_objects[obj_name]
            else:
                uid = ig_fixed_object_handles[obj_name].get_body_id()
                obj_info = scene_objects_fixed[obj_name]
            fake_perception.register_object(uid, obj_info, obj_name)
        pdm = PredicateDemonstrationManager(
            PATHS["data_dir"], fake_scene, fake_perception
        )

        # Get the defined number of demonstrations
        pos_samples_counter = 0
        neg_samples_counter = 0
        while (
            pos_samples_counter < samples_per_scene[scene_idx]
            or neg_samples_counter < samples_per_scene[scene_idx]
        ):
            # Rearrange objects
            positive_obj = np.random.choice(list(ig_object_handles.keys()))
            support_obj = np.random.choice(list(ig_fixed_object_handles.keys()))
            res_on = (
                ig_object_handles[positive_obj]
                .states[object_states.OnTop]
                .set_value(
                    ig_fixed_object_handles[support_obj],
                    True,
                    use_ray_casting_method=True,
                )
            )
            for other_obj in ig_object_handles.keys():
                if other_obj == positive_obj:
                    continue
                new_pos = np.random.uniform([-1.5, -1.5, 0.2], [1.5, 1.5, 2.0])
                new_orient = Rotation.random().as_quat()
                ig_object_handles[other_obj].set_position_orientation(
                    new_pos, new_orient
                )
            for _ in range(1200):
                p.stepSimulation()

            # Positive example
            if pos_samples_counter < samples_per_scene[scene_idx]:
                if predicates.on_(
                    ig_fixed_object_handles[support_obj].get_body_id(),
                    ig_object_handles[positive_obj].get_body_id(),
                    above_tol=0.05,
                ):
                    pdm.capture_demonstration(
                        "on_supporting_ig", [support_obj, positive_obj], label=True
                    )
                    pos_samples_counter += 1

            # Negative example
            if neg_samples_counter < samples_per_scene[scene_idx]:
                selected_obj = np.random.choice(list(ig_object_handles.keys()))
                if not predicates.on_(
                    ig_fixed_object_handles[support_obj].get_body_id(),
                    ig_object_handles[selected_obj].get_body_id(),
                    above_tol=0.05,
                ):
                    pdm.capture_demonstration(
                        "on_supporting_ig", [support_obj, selected_obj], label=False
                    )
                    neg_samples_counter += 1

        # Close simulator
        s.disconnect()
        del s
        del predicates


if __name__ == "__main__":
    main()
