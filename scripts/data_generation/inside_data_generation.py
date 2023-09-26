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

from scipy.spatial.transform import Rotation

from highlevel_planning_py.knowledge.predicates import PredicatesBase
from highlevel_planning_py.predicate_learning.demonstrations import (
    PredicateDemonstrationManager,
)
from highlevel_planning_py.sim.scene_base import SceneBase
from highlevel_planning_py.sim.fake_perception import FakePerceptionPipeline
from highlevel_planning_py.tools_pl.logger_stdout import LoggerStdout

from highlevel_planning_py.predicate_learning import training_utils as tu
from highlevel_planning_py.predicate_learning import data_gen_utils as dgu
from highlevel_planning_py.sim.world import WorldPybullet
from highlevel_planning_py.tools_pl.path import get_path_dict

PATHS_SELECTOR = "local"
NUM_SIM_STEPS = 500


@dataclass
class DataGenerationConfig(tu.ConfigBase):
    predicate_name: str
    num_samples: int
    num_samples_per_scene: int = 10
    scale_range_arg1: Tuple[float, float] = (0.8, 1.2)
    scale_range_arg2: Tuple[float, float] = (0.6, 1.5)

    num_supported_objects_range: Tuple[int, int] = (5, 13)
    num_supporting_objects: int = 2

    neg_distribution_labels: Tuple = (
        "obj_float",
        "obj_ground",
        "obj_float_above",
        "obj_sunken",
        "obj_inside_other",
        "obj_inside_collision",
        "obj_sticking_out",
    )
    neg_distribution: Tuple = (0.1, 0.1, 0.15, 0.15, 0.1, 0.2, 0.2)
    pos_distribution_labels: Tuple = ("obj_alone", "obj_multi")
    pos_distribution: Tuple = (0.2, 0.8)

    float_sunken_offset_range: Tuple[float, float] = (0.05, 0.2)


def generate_samples(
    config: DataGenerationConfig,
    available_objects,
    available_container_objects,
    logger,
    data_dir,
    data_session_id,
    demo_dir,
    paths,
):
    # Iterate over scenes
    num_negative_left = int(np.floor(config.num_samples / 2))
    num_positive_left = config.num_samples - num_negative_left
    sample_stats = defaultdict(int)

    start_time = time.time()
    time_elapsed = 0.0
    time_per_sample = 0.0

    while num_positive_left > 0 or num_negative_left > 0:
        time_elapsed = time.time() - start_time
        num_processed = config.num_samples - num_positive_left - num_negative_left
        if num_processed > 0:
            time_per_sample = time_elapsed / num_processed
        else:
            time_per_sample = 1
        print(
            f">>>>> Remaining: {num_positive_left} pos and {num_negative_left} neg. "
            f"Time left: {int((config.num_samples-num_processed)*time_per_sample/60)} min <<<<<"
        )

        # Set up simulator
        world = WorldPybullet(style="direct", sleep=False)
        predicates = PredicatesBase(world.client_id)
        p.setAdditionalSearchPath(
            pybullet_data.getDataPath(), physicsClientId=world.client_id
        )

        # Select supporting objects
        ig_fixed_object_handles, scene_objects_fixed = dgu.create_objects_pybullet(
            config.num_supporting_objects,
            available_container_objects,
            "fixed_obj",
            config.scale_range_arg1,
            paths,
            world,
        )

        # Place fixed objects in simulation
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
                pos_x = np.random.uniform(-1.5, 1.5)
                pos_y = np.random.uniform(-1.5, 1.5)
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
                ig_fixed_object_handles, world.client_id
            )

        # Select contained objects and scales
        num_objects = np.random.randint(*config.num_supported_objects_range)
        ig_object_handles, scene_objects = dgu.create_objects_pybullet(
            num_objects, available_objects, "obj", config.scale_range_arg2, paths, world
        )

        fake_world = dgu.FakeWorld(world.client_id)
        fake_scene = SceneBase(fake_world, {})
        fake_scene.set_objects({**scene_objects_fixed, **scene_objects})
        fake_perception = FakePerceptionPipeline(logger, world.client_id, paths)
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

            if positive_sample:
                label_pos = "pos"
                sample_type = np.random.choice(
                    config.pos_distribution_labels, p=config.pos_distribution
                )
            else:
                label_pos = "neg"
                sample_type = np.random.choice(
                    config.neg_distribution_labels, p=config.neg_distribution
                )

            arg_object_first = bool(np.random.randint(0, 2))

            sample_drawn = False
            max_retries = 30
            retries = 0
            while not sample_drawn and retries <= max_retries:
                retries += 1

                remaining_objects = list(ig_object_handles.keys())
                if positive_sample:
                    positive_obj = np.random.choice(remaining_objects)
                    remaining_objects.remove(positive_obj)
                    support_obj = np.random.choice(list(ig_fixed_object_handles.keys()))

                    if arg_object_first:
                        dgu.place_inside(
                            positive_obj,
                            support_obj,
                            ig_object_handles,
                            ig_fixed_object_handles,
                            fake_perception,
                        )

                    if sample_type == "obj_alone":
                        dgu.place_randomly(remaining_objects, ig_object_handles)
                    elif sample_type == "obj_multi":
                        num_other = np.random.randint(1, len(remaining_objects) + 1)
                        other_objs = np.random.choice(
                            remaining_objects, size=num_other, replace=False
                        )
                        remaining_objects = [
                            el for el in remaining_objects if el not in other_objs
                        ]
                        for obj in other_objs:
                            dgu.place_inside(
                                obj,
                                support_obj,
                                ig_object_handles,
                                ig_fixed_object_handles,
                                fake_perception,
                            )
                        dgu.place_randomly(remaining_objects, ig_object_handles)
                    else:
                        raise NotImplementedError

                    if not arg_object_first:
                        dgu.place_inside(
                            positive_obj,
                            support_obj,
                            ig_object_handles,
                            ig_fixed_object_handles,
                            fake_perception,
                        )

                    for _ in range(NUM_SIM_STEPS):
                        p.stepSimulation(physicsClientId=world.client_id)

                    # Record sample
                    if predicates.inside_oabb(
                        ig_fixed_object_handles[support_obj].get_body_id(),
                        ig_object_handles[positive_obj].get_body_id(),
                        fake_perception.get_object_oabb(support_obj),
                        fake_perception.get_object_oabb(positive_obj),
                        fake_perception.get_object_centroid(support_obj),
                        fake_perception.get_object_centroid(positive_obj),
                    ) and not dgu.check_in_collision(
                        {**ig_object_handles, **ig_fixed_object_handles},
                        world.client_id,
                        tolerance=0.005,
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
                        sample_drawn = True
                else:
                    negative_obj = np.random.choice(remaining_objects)
                    remaining_objects.remove(negative_obj)
                    remaining_support = list(ig_fixed_object_handles.keys())
                    support_obj = np.random.choice(remaining_support)
                    remaining_support.remove(support_obj)

                    dgu.place_randomly(remaining_objects, ig_object_handles)
                    for _ in range(NUM_SIM_STEPS):
                        p.stepSimulation(physicsClientId=world.client_id)

                    if sample_type == "obj_float":
                        dgu.place_randomly([negative_obj], ig_object_handles)
                    elif sample_type == "obj_ground":
                        dgu.place_randomly([negative_obj], ig_object_handles)
                        for _ in range(NUM_SIM_STEPS):
                            p.stepSimulation(physicsClientId=world.client_id)
                    elif sample_type == "obj_float_above":
                        dgu.place_inside(
                            negative_obj,
                            support_obj,
                            ig_object_handles,
                            ig_fixed_object_handles,
                            fake_perception,
                        )
                        for _ in range(NUM_SIM_STEPS):
                            p.stepSimulation(physicsClientId=world.client_id)
                        pos = ig_object_handles[negative_obj].get_position()
                        offset = np.random.uniform(*config.float_sunken_offset_range)
                        new_pos = (pos[0], pos[1], pos[2] + offset)
                        ig_object_handles[negative_obj].set_position(new_pos)
                    elif sample_type == "obj_sunken":
                        dgu.place_inside(
                            negative_obj,
                            support_obj,
                            ig_object_handles,
                            ig_fixed_object_handles,
                            fake_perception,
                        )
                        for _ in range(NUM_SIM_STEPS):
                            p.stepSimulation(physicsClientId=world.client_id)
                        pos = ig_object_handles[negative_obj].get_position()
                        offset = np.random.uniform(*config.float_sunken_offset_range)
                        new_pos = (pos[0], pos[1], pos[2] - offset)
                        ig_object_handles[negative_obj].set_position(new_pos)
                    elif sample_type == "obj_inside_other":
                        other_support = np.random.choice(remaining_support)
                        dgu.place_inside(
                            negative_obj,
                            other_support,
                            ig_object_handles,
                            ig_fixed_object_handles,
                            fake_perception,
                        )
                        for _ in range(NUM_SIM_STEPS):
                            p.stepSimulation(physicsClientId=world.client_id)

                        # Make sure object is actually inside other
                        if not predicates.inside_oabb(
                            ig_fixed_object_handles[other_support].get_body_id(),
                            ig_object_handles[negative_obj].get_body_id(),
                            fake_perception.get_object_oabb(other_support),
                            fake_perception.get_object_oabb(negative_obj),
                            fake_perception.get_object_centroid(other_support),
                            fake_perception.get_object_centroid(negative_obj),
                        ):
                            continue
                    elif sample_type == "obj_inside_collision":
                        dgu.place_inside(
                            negative_obj,
                            support_obj,
                            ig_object_handles,
                            ig_fixed_object_handles,
                            fake_perception,
                        )

                        for _ in range(NUM_SIM_STEPS):
                            p.stepSimulation(physicsClientId=world.client_id)

                        # Make sure negative object is inside container
                        if not predicates.inside_oabb(
                            ig_fixed_object_handles[support_obj].get_body_id(),
                            ig_object_handles[negative_obj].get_body_id(),
                            fake_perception.get_object_oabb(support_obj),
                            fake_perception.get_object_oabb(negative_obj),
                            fake_perception.get_object_centroid(support_obj),
                            fake_perception.get_object_centroid(negative_obj),
                        ):
                            continue

                        # Place other objects
                        num_other = np.random.randint(1, len(remaining_objects) + 1)
                        other_objs = np.random.choice(
                            remaining_objects, size=num_other, replace=False
                        )
                        for obj in other_objs:
                            dgu.place_inside(
                                obj,
                                support_obj,
                                ig_object_handles,
                                ig_fixed_object_handles,
                                fake_perception,
                            )

                        for _ in range(NUM_SIM_STEPS):
                            p.stepSimulation(physicsClientId=world.client_id)

                        # Move negative object into collision with first other object
                        collision_pos = ig_object_handles[other_objs[0]].get_position()
                        collision_oabb = fake_perception.get_object_oabb(other_objs[0])
                        collision_size = np.mean(collision_oabb[1, :2])
                        collision_offset = np.random.uniform(
                            -collision_size, collision_size, size=2
                        )
                        old_pos = ig_object_handles[negative_obj].get_position()
                        new_pos = (
                            collision_pos[0] + collision_offset[0],
                            collision_pos[1] + collision_offset[1],
                            old_pos[2],
                        )
                        ig_object_handles[negative_obj].set_position(new_pos)

                        # Make sure negative object is in collision with other object and not with container
                        if not dgu.check_in_collision_pair(
                            ig_object_handles[negative_obj].get_body_id(),
                            ig_object_handles[other_objs[0]].get_body_id(),
                            world.client_id,
                            tolerance=0.005,
                        ) or dgu.check_in_collision_pair(
                            ig_object_handles[negative_obj].get_body_id(),
                            ig_fixed_object_handles[support_obj].get_body_id(),
                            world.client_id,
                            tolerance=0.005,
                        ):
                            continue
                    elif sample_type == "obj_sticking_out":
                        if arg_object_first:
                            dgu.place_inside(
                                negative_obj,
                                support_obj,
                                ig_object_handles,
                                ig_fixed_object_handles,
                                fake_perception,
                            )

                        # Place other objects
                        num_other = np.random.randint(0, len(remaining_objects) + 1)
                        other_objs = np.random.choice(
                            remaining_objects, size=num_other, replace=False
                        )
                        for obj in other_objs:
                            dgu.place_inside(
                                obj,
                                support_obj,
                                ig_object_handles,
                                ig_fixed_object_handles,
                                fake_perception,
                            )

                        if not arg_object_first:
                            dgu.place_inside(
                                negative_obj,
                                support_obj,
                                ig_object_handles,
                                ig_fixed_object_handles,
                                fake_perception,
                            )

                        for _ in range(NUM_SIM_STEPS):
                            p.stepSimulation(physicsClientId=world.client_id)
                    else:
                        raise NotImplementedError

                    if (
                        sample_type == "obj_inside_collision"
                        or not predicates.inside_oabb(
                            ig_fixed_object_handles[support_obj].get_body_id(),
                            ig_object_handles[negative_obj].get_body_id(),
                            fake_perception.get_object_oabb(support_obj),
                            fake_perception.get_object_oabb(negative_obj),
                            fake_perception.get_object_centroid(support_obj),
                            fake_perception.get_object_centroid(negative_obj),
                        )
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
                        sample_drawn = True

        # Close simulator
        world.close()
        del world
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
    write_out = {
        "sample_stats": sample_stats,
        "pos_nominal": {
            label: 0.5 * config.num_samples * ratio
            for label, ratio in zip(
                config.pos_distribution_labels, config.pos_distribution
            )
        },
        "neg_nominal": {
            label: 0.5 * config.num_samples * ratio
            for label, ratio in zip(
                config.neg_distribution_labels, config.neg_distribution
            )
        },
        "time_elapsed": time_elapsed,
        "time_per_sample": time_per_sample,
    }
    time_string = time.strftime("%y%m%d-%H%M%S")
    filename = f"{time_string}_sample_stats.txt"
    filename_txt = os.path.join(demo_dir, filename)
    with open(filename_txt, "w") as f:
        pprint.pprint(write_out, f, sort_dicts=True)


def main():
    logger = LoggerStdout()
    paths = get_path_dict(PATHS_SELECTOR)
    data_dir = os.path.join(paths["data_dir"], "predicates", "data")
    dataset_id = "230925-155200_demonstrations_features"
    time_now = datetime.now()
    time_string = time_now.strftime("%y%m%d-%H%M%S")
    if dataset_id is None:
        dataset_id = f"{time_string}_demonstrations_features"

    # Find available YCB objects
    ycb_objects = dgu.find_ycb_objects()

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
    train_objects_container = [
        ("asset_dir", "parsed_xacros/container_no_lid_50_50.urdf"),
        ("asset_dir", "parsed_xacros/container_no_lid_60_40.urdf"),
    ]

    pred_name_train = "inside_drawer"
    config_train = DataGenerationConfig(
        device="cpu",
        predicate_name=pred_name_train,
        num_samples=20000,
        num_samples_per_scene=25,
        scale_range_arg1=(0.8, 1.2),
        scale_range_arg2=(0.5, 1.6),
        num_supported_objects_range=(5, 10),
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

    pred_name_test = "inside_drawer_test"
    config_test = DataGenerationConfig(
        device="cpu",
        predicate_name=pred_name_test,
        num_samples=2000,
        num_samples_per_scene=25,
        scale_range_arg1=(0.6, 1.4),
        scale_range_arg2=(0.4, 1.8),
        num_supported_objects_range=(5, 14),
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
        train_objects_container,
        logger,
        data_dir=data_dir,
        data_session_id=dataset_id,
        demo_dir=demo_dir_train,
        paths=paths,
    )
    generate_samples(
        config_test,
        test_objects,
        train_objects_container,
        logger,
        data_dir=data_dir,
        data_session_id=dataset_id,
        demo_dir=demo_dir_test,
        paths=paths,
    )


if __name__ == "__main__":
    main()
