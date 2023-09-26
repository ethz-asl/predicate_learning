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
from datetime import datetime
import numpy as np
import pybullet as p
import igibson
from igibson.simulator import Simulator
from igibson.scenes.empty_scene import EmptyScene
from igibson.objects.articulated_object import RBOObject

from highlevel_planning_py.tools_pl.logger_stdout import LoggerStdout
from highlevel_planning_py.tools_pl.util import ObjectInfo, capture_image_pybullet
from highlevel_planning_py.sim.cupboard import get_cupboard_info
from highlevel_planning_py.sim.scene_base import SceneBase
from highlevel_planning_py.predicate_learning.features import PredicateFeatureManager
from highlevel_planning_py.predicate_learning.visualization_utils import visualize_aabb

import igibson_on_data_generation as idg


SRCROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
PATHS = {
    "data_dir": os.path.join(os.path.expanduser("~"), "Data", "highlevel_planning"),
    "asset_dir": os.path.join(SRCROOT, "data", "models"),
    "igibson_dir": igibson.assets_path,
}


def visualize_object(obj, out_dir, logger):
    scale = 1.0
    if "ycb" in obj:
        obj_handle = idg.YCBObjectWrapper(obj.split("/")[-1], scale=scale)
        obj_info = ObjectInfo(
            urdf_name_=obj.split("/")[-1],
            urdf_path_=obj,
            init_pos_=np.array([0, 0, 0]),
            init_orient_=np.array([0, 0, 0, 1]),
            init_scale_=scale,
        )
    elif "rbo" in obj:
        obj_handle = RBOObject(obj.split("/")[-1], scale=scale)
        obj_info = ObjectInfo(
            urdf_name_=obj.split("/")[-1],
            urdf_path_=obj_handle.filename,
            init_pos_=np.array([0, 0, 0]),
            init_orient_=np.array([0, 0, 0, 1]),
            init_scale_=scale,
        )
    else:
        obj_handle = idg.ArticulatedObjectWrapper(
            obj, scale=scale, merge_fixed_links=True
        )
        obj_info = ObjectInfo(
            urdf_path_=obj,
            init_pos_=np.array([0, 0, 0]),
            init_orient_=np.array([0, 0, 0, 1]),
            init_scale_=scale,
            merge_fixed_links_=True,
        )

    s = Simulator(mode="gui")
    scene = EmptyScene()
    s.import_scene(scene)
    for floor_id in scene.floor_body_ids:
        p.removeBody(floor_id, physicsClientId=s.cid)

    if s.mode == "gui":
        p.removeBody(s.viewer.constraint_marker.body_id, physicsClientId=s.cid)
        p.removeBody(s.viewer.constraint_marker2.body_id, physicsClientId=s.cid)

    s.import_object(obj_handle)
    obj_handle.set_position_orientation([0, 0, 0], [0, 0, 0, 1])

    # Capture image without AABB
    filename = os.path.join(out_dir, f"{obj_info.urdf_name.split('/')[-1]}__nobb.png")
    capture_image_pybullet(
        s.cid, path=filename, show=False, camera_pos=(0.25, 0.25, 0.25)
    )

    fake_world = idg.FakeWorld(s.cid)
    fake_scene = SceneBase(fake_world, {})
    fake_scene.set_objects({"obj0": obj_info})
    fake_perception = idg.FakePerceptionPipeline(logger, s.cid, PATHS)
    for obj_name in fake_scene.objects:
        uid = idg.get_body_id(obj_handle)
        fake_perception.register_object(uid, obj_info, obj_name)
    pfm = PredicateFeatureManager("", None, None, None, None, logger, PATHS)
    feature_dict = pfm.get_features("obj0", fake_perception)
    features = np.array([list(feature_dict.values())])
    visualize_aabb(features[:, 3:9], s.cid)

    # Capture image without AABB
    # filename = os.path.join(out_dir, f"{obj_info.urdf_name}__aabb.png")
    # capture_image_pybullet(s.cid, show=True, camera_pos=(0.3, 0.3, 0.3))

    # Wait before advancing
    _ = input("Enter to continue...")

    s.disconnect()


def main():
    logger = LoggerStdout()

    time_string = datetime.now().strftime("%y%m%d-%H%M%S")
    out_dir = os.path.join(
        PATHS["data_dir"], "predicates", "data", "objects", time_string
    )
    os.makedirs(out_dir)

    # Find available YCB objects
    ycb_dir = os.path.join(igibson.assets_path, "models", "ycb")
    available_objects_ = os.listdir(ycb_dir)
    available_objects_.sort()
    ycb_objects = list()
    for obj in available_objects_:
        path = os.path.join(ycb_dir, obj)
        if os.path.isdir(path):
            ycb_objects.append(path)
    del available_objects_
    assert len(ycb_objects) == 21

    # Select own objects to use
    own_objects = [
        "cube_small.urdf",
        "lego/lego.urdf",
        "duck_vhacd.urdf",
        os.path.join(PATHS["asset_dir"], "tall_box.urdf"),
    ]

    # Split into train and test objects
    train_objects = ycb_objects[:15]
    train_objects.extend(own_objects[:2])
    test_objects = ycb_objects[15:]
    test_objects.extend(own_objects[2:])

    # Set fixed objects
    scene_objects_fixed_train = dict()
    cabinet_lower_path = os.path.join(
        igibson.assets_path, "models/cabinet2/cabinet_0007.urdf"
    )
    scene_objects_fixed_train["cabinet_lower"] = ObjectInfo(
        urdf_name_="cabinet_0007.urdf",
        urdf_path_=cabinet_lower_path,
        init_pos_=np.array([-0.5, 0, 0.5]),
        init_orient_=np.array([0, 0, 0, 1]),
        merge_fixed_links_=True,
    )
    cabinet_upper_path = os.path.join(
        igibson.assets_path, "models/cabinet/cabinet_0004.urdf"
    )
    scene_objects_fixed_train["cabinet_upper"] = ObjectInfo(
        urdf_name_="cabinet_0004.urdf",
        urdf_path_=cabinet_upper_path,
        init_pos_=np.array([1.0, 0, 0.5]),
        init_orient_=np.array([0, 0, 0, 1]),
        merge_fixed_links_=True,
    )

    scene_objects_fixed_test = dict()
    scene_objects_fixed_test["table"] = ObjectInfo(
        urdf_name_="table/table.urdf",
        urdf_path_=os.path.join(PATHS["asset_dir"], "table", "table.urdf"),
        init_pos_=np.array([3.0, 0.0, 0.0]),
        init_orient_=np.array([0.0, 0.0, 0.0, 1.0]),
    )
    scene_objects_fixed_test["cupboard"] = get_cupboard_info(
        PATHS["asset_dir"], pos=[0.0, 1.5, 0.0], orient=[0.0, 0.0, 0.0, 1.0], scale=1.0
    )

    # for obj in train_objects:
    #     visualize_object(obj, out_dir, logger)
    for obj in test_objects[6:]:
        visualize_object(obj, out_dir, logger)


if __name__ == "__main__":
    main()
