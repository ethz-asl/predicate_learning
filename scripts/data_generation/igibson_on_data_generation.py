from typing import Tuple
from datetime import datetime
import numpy as np
from dataclasses import dataclass
import os
from collections import defaultdict
import time
import pprint

import pybullet as p
import pybullet_data

from scipy.spatial.transform.rotation import Rotation

import igibson
from igibson.simulator import Simulator
from igibson.scenes.empty_scene import EmptyScene

from highlevel_planning_py.knowledge.predicates import PredicatesBase
from highlevel_planning_py.predicate_learning.demonstrations import (
    PredicateDemonstrationManager,
)
from highlevel_planning_py.sim.scene_base import SceneBase
from highlevel_planning_py.sim.fake_perception import FakePerceptionPipeline
from highlevel_planning_py.tools_pl.logger_stdout import LoggerStdout

# from highlevel_planning_py.sim.cupboard import get_cupboard_info
from highlevel_planning_py.predicate_learning import training_utils as tu
from highlevel_planning_py.predicate_learning import data_gen_utils as dgu


SRCROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
PATHS = {
    "": "",
    "data_dir": os.path.join(os.path.expanduser("~"), "Data", "highlevel_planning"),
    "asset_dir": os.path.join(SRCROOT, "data", "models"),
    "igibson_dir": igibson.assets_path,
}


@dataclass
class DataGenerationConfig(tu.ConfigBase):
    predicate_name: str
    num_samples: int
    num_samples_per_scene: int = 10
    scale_range_arg1: Tuple[float, float] = (0.8, 1.2)
    scale_range_arg2: Tuple[float, float] = (0.6, 1.5)
    test_above_tolerance: float = 0.03

    num_supported_objects_range: Tuple[int, int] = (5, 9)
    num_supporting_objects: int = 2

    neg_distribution_labels: Tuple = (
        "obj_float",
        "obj_ground",
        "obj_float_above",
        "obj_sunken",
        "obj_inside",
        "obj_on_other",
    )
    neg_distribution: Tuple = (0.15, 0.2, 0.15, 0.15, 0.25, 0.1)
    pos_distribution_labels: Tuple = ("obj_alone", "obj_multi")
    pos_distribution: Tuple = (0.5, 0.5)

    float_sunken_offset_range: Tuple[float, float] = (0.1, 0.25)
    inside_offset_range: Tuple[float, float] = (0.45, 0.65)


def generate_samples(
    config: DataGenerationConfig,
    available_objects,
    available_supporting_objects,
    logger,
    data_dir,
    data_session_id,
    demo_dir,
):
    # Iterate over scenes
    num_negative_left = int(np.floor(config.num_samples / 2))
    num_positive_left = config.num_samples - num_negative_left
    sample_stats = defaultdict(int)
    while num_positive_left > 0 or num_negative_left > 0:
        print(
            f">>>>> Remaining: {num_positive_left} pos and {num_negative_left} neg <<<<<"
        )

        # Set up simulator
        s = Simulator(mode="headless")
        # s = Simulator(mode="gui")
        scene = EmptyScene()
        predicates = PredicatesBase(s.cid)
        s.import_scene(scene)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=s.cid)
        if s.mode == "gui":
            p.removeBody(s.viewer.constraint_marker.body_id, physicsClientId=s.cid)
            p.removeBody(s.viewer.constraint_marker2.body_id, physicsClientId=s.cid)

        # Select supported objects and scales
        num_objects = np.random.randint(*config.num_supported_objects_range)
        ig_object_handles, scene_objects = dgu.create_objects(
            num_objects, available_objects, "obj", config.scale_range_arg2, PATHS
        )

        # Select supporting objects
        ig_fixed_object_handles, scene_objects_fixed = dgu.create_objects(
            config.num_supporting_objects,
            available_supporting_objects,
            "fixed_obj",
            config.scale_range_arg1,
            PATHS,
        )

        # Import into simulation
        for handle in ig_fixed_object_handles.values():
            s.import_object(handle)

        fixed_objects_in_collision = True
        failure_count = -1
        while fixed_objects_in_collision:
            failure_count += 1
            if failure_count > 20:
                raise RuntimeError(
                    "Too many failures sampling collision-free fixed object poses"
                )
            for obj_name, obj in ig_fixed_object_handles.items():
                # Sample position and orientation
                pos_x = np.random.uniform(-1.8, 1.8)
                pos_y = np.random.uniform(-1.8, 1.8)
                pos_z = (
                    0.5 if "cabinet" in scene_objects_fixed[obj_name].urdf_name else 0.0
                )  # TODO could adjust this to individual object height
                orient_z = np.random.uniform(-180, 180)
                orient_quat = Rotation.from_euler("z", orient_z, degrees=True)
                obj.set_position_orientation(
                    [pos_x, pos_y, pos_z], orient_quat.as_quat()
                )
            # Check if they are collision-free
            fixed_objects_in_collision = dgu.check_in_collision(
                ig_fixed_object_handles, s.cid
            )

        for handle in ig_object_handles.values():
            s.import_object(handle)

        fake_world = dgu.FakeWorld(s.cid)
        fake_scene = SceneBase(fake_world, {})
        fake_scene.set_objects({**scene_objects_fixed, **scene_objects})
        fake_perception = FakePerceptionPipeline(logger, s.cid, PATHS)
        for obj_name in fake_scene.objects:
            if obj_name in ig_object_handles:
                uid = dgu.get_body_id(ig_object_handles[obj_name])
                obj_info = scene_objects[obj_name]
            else:
                uid = dgu.get_body_id(ig_fixed_object_handles[obj_name])
                obj_info = scene_objects_fixed[obj_name]
            fake_perception.register_object(uid, obj_info, obj_name)
        pdm = PredicateDemonstrationManager(
            data_dir, data_session_id, fake_scene, fake_perception
        )

        samples_this_scene = 0
        while samples_this_scene < config.num_samples_per_scene and (
            num_positive_left > 0 or num_negative_left > 0
        ):
            # Determine whether to draw a positive or negative sample
            if num_positive_left > 0 and num_negative_left > 0:
                positive_sample = bool(np.random.randint(0, 2))
            elif num_positive_left > 0:
                positive_sample = True
            else:
                positive_sample = False

            label_pos = "pos" if positive_sample else "neg"

            remaining_objects = list(ig_object_handles.keys())
            if positive_sample:
                positive_obj = np.random.choice(remaining_objects)
                remaining_objects.remove(positive_obj)
                support_obj = np.random.choice(list(ig_fixed_object_handles.keys()))
                dgu.place_on(
                    positive_obj,
                    support_obj,
                    ig_object_handles,
                    ig_fixed_object_handles,
                )

                sample_type = np.random.choice(
                    config.pos_distribution_labels, p=config.pos_distribution
                )
                if sample_type == "obj_alone":
                    dgu.place_randomly(remaining_objects, ig_object_handles)
                elif sample_type == "obj_multi":
                    other_obj = np.random.choice(remaining_objects)
                    remaining_objects.remove(other_obj)
                    dgu.place_on(
                        other_obj,
                        support_obj,
                        ig_object_handles,
                        ig_fixed_object_handles,
                    )
                    dgu.place_randomly(remaining_objects, ig_object_handles)
                for _ in range(1200):
                    p.stepSimulation()

                # Record sample
                if predicates.on_(
                    ig_fixed_object_handles[support_obj].get_body_id(),
                    ig_object_handles[positive_obj].get_body_id(),
                    above_tol=config.test_above_tolerance,
                ):
                    pdm.capture_demonstration(
                        config.predicate_name,
                        [support_obj, positive_obj],
                        label=True,
                        comment=sample_type,
                    )
                    num_positive_left -= 1
                    samples_this_scene += 1
                    sample_stats[f"{label_pos}-{sample_type}"] += 1
            else:
                negative_obj = np.random.choice(remaining_objects)
                remaining_objects.remove(negative_obj)
                remaining_support = list(ig_fixed_object_handles.keys())
                support_obj = np.random.choice(remaining_support)
                remaining_support.remove(support_obj)

                sample_type = np.random.choice(
                    config.neg_distribution_labels, p=config.neg_distribution
                )

                dgu.place_randomly(remaining_objects, ig_object_handles)
                for _ in range(1200):
                    p.stepSimulation()

                if sample_type == "obj_float":
                    dgu.place_randomly([negative_obj], ig_object_handles)
                elif sample_type == "obj_ground":
                    dgu.place_randomly([negative_obj], ig_object_handles)
                    for _ in range(1200):
                        p.stepSimulation()
                elif sample_type == "obj_float_above":
                    dgu.place_on(
                        negative_obj,
                        support_obj,
                        ig_object_handles,
                        ig_fixed_object_handles,
                    )
                    pos = ig_object_handles[negative_obj].get_position()
                    offset = np.random.uniform(*config.float_sunken_offset_range)
                    new_pos = (pos[0], pos[1], pos[2] + offset)
                    ig_object_handles[negative_obj].set_position(new_pos)
                elif sample_type == "obj_sunken":
                    dgu.place_on(
                        negative_obj,
                        support_obj,
                        ig_object_handles,
                        ig_fixed_object_handles,
                    )
                    pos = ig_object_handles[negative_obj].get_position()
                    offset = np.random.uniform(*config.float_sunken_offset_range)
                    new_pos = (pos[0], pos[1], pos[2] - offset)
                    ig_object_handles[negative_obj].set_position(new_pos)
                elif sample_type == "obj_inside":
                    dgu.place_on(
                        negative_obj,
                        support_obj,
                        ig_object_handles,
                        ig_fixed_object_handles,
                    )
                    pos = ig_object_handles[negative_obj].get_position()
                    offset = np.random.uniform(*config.inside_offset_range)
                    new_pos = (pos[0], pos[1], pos[2] - offset)
                    ig_object_handles[negative_obj].set_position(new_pos)
                elif sample_type == "obj_on_other":
                    other_support = np.random.choice(remaining_support)
                    dgu.place_on(
                        negative_obj,
                        other_support,
                        ig_object_handles,
                        ig_fixed_object_handles,
                    )

                if not predicates.on_(
                    ig_fixed_object_handles[support_obj].get_body_id(),
                    ig_object_handles[negative_obj].get_body_id(),
                    above_tol=config.test_above_tolerance,
                ):
                    pdm.capture_demonstration(
                        config.predicate_name,
                        [support_obj, negative_obj],
                        label=False,
                        comment=sample_type,
                    )
                    num_negative_left -= 1
                    samples_this_scene += 1
                    sample_stats[f"{label_pos}-{sample_type}"] += 1

        # Close simulator
        s.disconnect()
        del s
        del predicates

    # Save stats data
    pos_total = 0
    neg_total = 0
    for label in sample_stats:
        if "pos" in label:
            pos_total += sample_stats[label]
        else:
            neg_total += sample_stats[label]
    sample_stats["pos_total"] = pos_total
    sample_stats["neg_total"] = neg_total
    time_string = time.strftime("%y%m%d-%H%M%S")
    filename = f"{time_string}_sample_stats.txt"
    filename_txt = os.path.join(demo_dir, filename)
    with open(filename_txt, "w") as f:
        pprint.pprint(sample_stats, f, sort_dicts=True)


def main():
    logger = LoggerStdout()
    data_dir = os.path.join(PATHS["data_dir"], "predicates", "data")
    dataset_id = "220831-175353_demonstrations_features"
    time_now = datetime.now()
    time_string = time_now.strftime("%y%m%d-%H%M%S")
    if dataset_id is None:
        dataset_id = f"{time_string}_demonstrations_features"

    # Find available YCB objects
    ycb_objects = dgu.find_ycb_objects()

    # Select RBO objects to use
    # rbo_dir = os.path.join(igibson.assets_path, "models", "rbo")
    # rbo_object_names = ["book", "pliers", "rubikscube", "treasurebox"]
    # rbo_objects = list()
    # for obj in rbo_object_names:
    #     rbo_objects.append(os.path.join(rbo_dir, obj))

    # Select own objects to use
    own_objects = [
        ("", "cube_small.urdf"),
        ("", "lego/lego.urdf"),
        ("", "duck_vhacd.urdf"),
        ("asset_dir", "tall_box.urdf"),
    ]

    # Split into train and test objects
    train_objects = ycb_objects[:15]
    train_objects.extend(own_objects[:2])
    test_objects = ycb_objects[15:]
    test_objects.extend(own_objects[2:])

    # Set fixed objects
    train_objects_supporting = [
        ("igibson_dir", "models/cabinet2/cabinet_0007.urdf"),
        ("igibson_dir", "models/cabinet/cabinet_0004.urdf"),
        ("asset_dir", "solid_support.urdf"),
    ]
    test_objects_supporting = [
        ("asset_dir", "table/table.urdf"),
        ("asset_dir", "parsed_xacros/cupboard2.urdf"),
    ]

    pred_name_train = "on_supporting_ig"
    config_train = DataGenerationConfig(
        device="cpu",
        predicate_name=pred_name_train,
        num_samples=20000,
        num_samples_per_scene=25,
        scale_range_arg1=(0.8, 1.2),
        scale_range_arg2=(0.5, 1.6),
    )
    demo_dir_train = os.path.join(
        data_dir, pred_name_train, "demonstrations", dataset_id
    )
    os.makedirs(demo_dir_train, exist_ok=True)
    tu.save_parameters(
        config_train.to_dict(),
        f"{time_string}_parameters_train",
        demo_dir_train,
        txt_only=True,
    )

    pred_name_test = "on_supporting_ig_test"
    config_test = DataGenerationConfig(
        device="cpu",
        predicate_name=pred_name_test,
        num_samples=2000,
        num_samples_per_scene=25,
        scale_range_arg1=(0.6, 1.4),
        scale_range_arg2=(0.4, 1.8),
    )
    demo_dir_test = os.path.join(data_dir, pred_name_test, "demonstrations", dataset_id)
    os.makedirs(demo_dir_test, exist_ok=True)
    tu.save_parameters(
        config_test.to_dict(),
        f"{time_string}_parameters_test",
        demo_dir_test,
        txt_only=True,
    )

    generate_samples(
        config_train,
        train_objects,
        train_objects_supporting,
        logger,
        data_dir=data_dir,
        data_session_id=dataset_id,
        demo_dir=demo_dir_train,
    )
    generate_samples(
        config_test,
        test_objects,
        test_objects_supporting,
        logger,
        data_dir=data_dir,
        data_session_id=dataset_id,
        demo_dir=demo_dir_test,
    )


if __name__ == "__main__":
    main()
