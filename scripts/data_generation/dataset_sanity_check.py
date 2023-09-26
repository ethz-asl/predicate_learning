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
import pickle
from collections import defaultdict
import torch
import time
from tqdm import tqdm

from highlevel_planning_py.predicate_learning import training_utils as tu
from highlevel_planning_py.predicate_learning.dataset import PredicateGraphDataset
from highlevel_planning_py.sim.world import WorldPybullet
from highlevel_planning_py.sim.scene_base import SceneBase
from highlevel_planning_py.sim.fake_perception import FakePerceptionPipeline
from highlevel_planning_py.tools_pl.logger_stdout import LoggerStdout
from highlevel_planning_py.predicate_learning.visualization_utils import visualize_aabb
from highlevel_planning_py.predicate_learning.groundtruths_inside import (
    # ManualClassifier_Inside_OABB,
    ManualClassifier_Inside_OABB_TrueGeom,
)

SRCROOT = os.path.join(os.path.dirname(__file__), "..", "..")
PATHS = {
    "": "",
    "data_dir": os.path.join(os.path.expanduser("~"), "Data", "highlevel_planning"),
    "asset_dir": os.path.join(SRCROOT, "data", "models"),
    "igibson_dir": os.path.join(os.path.expanduser("~"), "Data", "igibson", "assets"),
}

DEBUG = False


def main():
    device = "cpu"

    dataset_id = "230606-124817_demonstrations_features"
    predicate_name = "inside_drawer"
    data_normalization = "none"

    predicate_dir = os.path.join(PATHS["data_dir"], "predicates")
    dataset_dir = os.path.join(predicate_dir, "data")
    demo_dir = os.path.join(dataset_dir, predicate_name, "demonstrations", dataset_id)

    time_now = datetime.now()
    time_string = time_now.strftime("%y%m%d_%H%M%S")
    out_dir = f"/tmp/{time_string}_dataset_sanity_check"
    os.makedirs(out_dir, exist_ok=False)

    dataset_class_config = tu.DatasetConfig(
        device,
        dataset_dir,
        dataset_id,
        predicate_name,
        {"train": 1.0},
        target_model="class",
        normalization_method=data_normalization,
        positive_only=False,
        include_surrounding=True,
        label_arg_objects=False,
        feature_version="v1",
        dataset_size=-1,
    )
    dataset = PredicateGraphDataset(dataset_class_config)

    if DEBUG:
        logger = LoggerStdout()
        world = WorldPybullet("gui", sleep=False)
        scene = SceneBase(world, PATHS, restored_objects=dict())
        perception = FakePerceptionPipeline(logger, world.client_id, PATHS)
        perception.populate_from_scene(scene)
    else:
        world = WorldPybullet("direct", sleep=False)

    # gt = ManualClassifier_Inside_OABB(device, "v1", pen_tolerance=0.0, world=world)
    gt = ManualClassifier_Inside_OABB_TrueGeom(
        device,
        "v1",
        pen_tolerance=0.001,
        paths=PATHS,
        data_session_id=dataset_id,
        debug=False,
    )

    counts = {
        "agree_pos": 0,
        "disagree_pos": 0,
        "agree_neg": 0,
        "disagree_neg": 0,
        "reason_disagree_pos": defaultdict(int),
        "reason_disagree_neg": defaultdict(int),
    }

    start_time = time.time()

    for i in tqdm(range(len(dataset))):
        features_args, features_others, label_ds, demo_id, obj_names = dataset.get_single(
            i, use_tensors=True
        )

        # Classify manually
        features_all = torch.cat((features_args, features_others), dim=0).unsqueeze(0)
        mask = torch.ones(features_all.size(0), features_all.size(1))
        demo_ids = [demo_id] * features_all.size(0)
        label_gt, reason = gt.check_reason(features_all, mask, demo_ids)
        label_gt = bool(label_gt.item())

        if label_gt == bool(label_ds):
            counts[f"agree_{'pos' if bool(label_ds) else 'neg'}"] += 1
        else:
            counts[f"disagree_{'pos' if bool(label_ds) else 'neg'}"] += 1
            counts[f"reason_disagree_{'pos' if bool(label_ds) else 'neg'}"][
                reason[0]
            ] += 1

            if not DEBUG:
                continue

            # Load demo
            meta_file_name = os.path.join(demo_dir, demo_id, "demo.pkl")
            with open(meta_file_name, "rb") as f:
                demo_meta_data = pickle.load(f)
            arguments, label, objects, sample_type = demo_meta_data

            assert label == bool(label_ds)

            # Visualize scene
            if objects != scene.objects:
                world.reset()
                scene.set_objects(objects)
                scene.add_objects(force_load=True)
                perception.reset()
                perception.populate_from_scene(scene)
                scene.show_object_labels()
            simstate_file_name = os.path.join(demo_dir, demo_id, "state.bullet")
            world.restore_state_file(simstate_file_name)

            visualize_aabb(features_args[:, 3:9].numpy(), world.client_id)

            gt.check_reason(features_all, mask, demo_ids)

            # input("Press enter to continue")

        # logger.info(
        #     f"Demo {demo_id}. Label: {label}, arguments: {arguments}, sample type: {sample_type}"
        # )

    end_time = time.time()
    print(
        f"Time for {len(dataset)} demos: {end_time - start_time} seconds. Classifier total time: {gt.total_time}."
    )

    counts["reason_disagree_neg"] = dict(counts["reason_disagree_neg"])
    counts["reason_disagree_pos"] = dict(counts["reason_disagree_pos"])
    print(counts)
    world.close()


if __name__ == "__main__":
    main()
