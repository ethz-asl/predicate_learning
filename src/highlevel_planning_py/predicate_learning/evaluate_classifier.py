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
import pickle
import time
from tqdm import tqdm
import logging
import pandas as pd
import torch
import torch_geometric as pyg

import highlevel_planning_py.predicate_learning.training_utils as tu
from highlevel_planning_py.predicate_learning.models_hybrid import (
    GANdiscriminatorHybrid,
)
from highlevel_planning_py.predicate_learning.models_sklearn import (
    DecisionTreeClassifier,
)
from highlevel_planning_py.predicate_learning.dataset_pointcloud import (
    PredicateEncodingGraphDataset,
)
from highlevel_planning_py.predicate_learning.dataset import PredicateGraphDataset
from highlevel_planning_py.tools_pl import util

SRCROOT = "/home/fjulian/Code/ros_manipulation_ws/src/high_level_planning_private/highlevel_planning"
PATHS = {
    "": "",
    "data_dir": os.path.join(os.path.expanduser("~"), "Data", "highlevel_planning"),
    "asset_dir": os.path.join(SRCROOT, "data", "models"),
    "igibson_dir": os.path.join(os.path.expanduser("~"), "Data", "igibson", "assets"),
}


def evaluate_class_run(
    paths: dict,
    training_id: str,
    dataset_name: str,
    weights_idx: int = -1,
    create_plots: bool = False,
    random_seed: int = 12,
    cuda: bool = False,
    training_type: str = "gan",
    dataset_size: int = -1,
    training_dir: str = None,
    timing_only: bool = False,
):
    assert training_type in [
        "gan",
        "pcgan",
        "decisiontree",
    ], "Not implemented for this training type"

    if create_plots:
        logging.info("Creating plots is not supported for this script. Ignoring flag.")

    # Paths
    predicate_dir = os.path.join(paths["data_dir"], "predicates")
    if training_dir is None:
        training_dir = os.path.join(predicate_dir, "training", training_id)
    else:
        training_dir = os.path.join(training_dir, training_id)
    dataset_dir = os.path.join(predicate_dir, "data")

    out_dir = os.path.join(training_dir, "post_eval")
    os.makedirs(out_dir, exist_ok=True)

    # Parameters
    run_config = tu.load_parameters(training_dir, "parameters_training")
    dataset_id = run_config.dataset_id
    device = "cuda" if cuda else "cpu"

    # Dataset
    if dataset_name == "detect":
        dataset_name = run_config.predicate_name
    elif dataset_name == "detect_test":
        dataset_name = f"{run_config.predicate_name}_test"

    logging.info(f"Evaluating run {training_id} on {dataset_name}...")

    demonstration_dir = os.path.join(
        dataset_dir, dataset_name, "demonstrations", dataset_id
    )

    pyg.seed_everything(random_seed)

    # Determine weights to use
    if training_type != "decisiontree":
        available_class_weights = util.get_weights_list(training_dir, "class")
        class_weights_file = available_class_weights[weights_idx]
        mdl_params = tu.load_parameters(training_dir, "parameters_class")
        mdl_params.device = device
        state_file = os.path.join(training_dir, "models", class_weights_file)
        mdl_state = torch.load(state_file)
        feature_version = (
            mdl_params.feature_version
            if hasattr(mdl_params, "feature_version")
            else run_config.dataset_feature_version
        )
    else:
        class_weights_file = "classifier_final.pkl"
        mdl_state = os.path.join(training_dir, class_weights_file)
        mdl_params = None
        feature_version = run_config.feature_version

    if timing_only:
        out_file = os.path.join(out_dir, "timing_classifier.csv")
        if os.path.isfile(out_file):
            logging.info("Already computed timing. Skipping.")
            return
        score_data = pd.DataFrame()
    else:
        out_file = os.path.join(out_dir, "scores_classifier.csv")
        if os.path.isfile(out_file):
            score_data = pd.read_csv(out_file)

            # Check if score was already computed
            mask = (
                (score_data["dataset"] == dataset_name)
                & (score_data["weights_file"] == class_weights_file)
                & (score_data["random_seed"] == random_seed)
            )
            find_scores = score_data[mask]
            if len(find_scores) > 0:
                logging.info(
                    f"Already computed scores for {dataset_name}, "
                    f"{class_weights_file} and seed {random_seed}. Skipping."
                )
                return
        else:
            score_data = pd.DataFrame()

    # Load class model
    logging.info(f"Using weights {class_weights_file}")
    if training_type == "gan":
        if run_config.model_type == "hybrid":
            model_class = GANdiscriminatorHybrid(mdl_params)
            dataset_type = PredicateGraphDataset
            include_surrounding = True
        else:
            raise NotImplementedError
    elif training_type == "pcgan":
        if run_config.model_type == "hybrid":
            model_class = GANdiscriminatorHybrid(mdl_params)
            dataset_type = PredicateEncodingGraphDataset
            include_surrounding = True
        else:
            raise NotImplementedError
    elif training_type == "decisiontree":
        model_class = DecisionTreeClassifier(mdl_params)
        dataset_type = (
            PredicateGraphDataset
            if run_config.feature_type == "manual"
            else PredicateEncodingGraphDataset
        )
        include_surrounding = False
    else:
        raise NotImplementedError
    model_class.load_state_dict(mdl_state)
    model_class.to(device)

    # Create dataset
    train_dataset_config = tu.load_parameters(
        training_dir, "parameters_data_class_train_val"
    )
    dataset_config = tu.DatasetConfig(
        device,
        dataset_dir,
        dataset_id,
        dataset_name,
        {"train": 1.0},
        target_model="class",
        normalization_method=train_dataset_config.normalization_method,
        positive_only=False,
        include_surrounding=include_surrounding,
        feature_version=feature_version,
        encoder_id=train_dataset_config.encoder_id,
        dataset_size=dataset_size,
    )
    dataset = dataset_type(dataset_config)

    # Loop through dataset, visualizing everything for every iteration
    if dataset_size == -1:
        len_dataset = len(dataset)
    else:
        len_dataset = min(len(dataset), dataset_size)

    if "on_clutter" in dataset_name:
        sample_types_neg = (
            "obj_float",
            "obj_ground",
            "obj_float_above",
            "obj_sunken",
            "obj_inside",
            "obj_on_other",
            "obj_on_collision",
        )
        sample_types_pos = ("obj_alone", "obj_multi")
    elif "inside_drawer" in dataset_name:
        sample_types_neg = (
            "obj_float",
            "obj_ground",
            "obj_float_above",
            "obj_sunken",
            "obj_inside_other",
            "obj_inside_collision",
            "obj_sticking_out",
        )
        sample_types_pos = ("obj_alone", "obj_multi")
    else:
        raise NotImplementedError
    counts = {"count_correct": 0, "count_incorrect": 0}
    for sample_type in sample_types_neg:
        counts[f"count_correct_neg-{sample_type}"] = 0
        counts[f"count_incorrect_neg-{sample_type}"] = 0
    for sample_type in sample_types_pos:
        counts[f"count_correct_pos-{sample_type}"] = 0
        counts[f"count_incorrect_pos-{sample_type}"] = 0

    start_time = time.perf_counter()

    timing = {"time": 0.0, "iterations": 0, "batch_size": 1}

    for i in tqdm(range(len_dataset)):
        batch = dataset[i]

        # Classify
        it_start_time = time.perf_counter()
        class_output = model_class.forward(batch)[0][0]
        if training_type == "decisiontree":
            pred_score = class_output
        else:
            pred_score = torch.sigmoid(class_output)
        timing["time"] += time.perf_counter() - it_start_time
        timing["iterations"] += 1

        if timing_only:
            continue

        pred_label = torch.round(pred_score).long().detach().cpu().item()
        groundtruth_label = batch.y[0].long().detach().cpu().item()

        # Get sample type
        demo_id = batch["demo_id"]
        demo_file = os.path.join(demonstration_dir, demo_id, "demo.pkl")
        with open(demo_file, "rb") as f:
            demo_data = pickle.load(f)
        sample_type = demo_data[3]

        # Update counts
        if groundtruth_label == pred_label:
            # Correctly classified
            counts["count_correct"] += 1
            if groundtruth_label == 1:
                counts[f"count_correct_pos-{sample_type}"] += 1
            else:
                counts[f"count_correct_neg-{sample_type}"] += 1
        else:
            # Incorrectly classified
            counts["count_incorrect"] += 1
            if groundtruth_label == 1:
                counts[f"count_incorrect_pos-{sample_type}"] += 1
            else:
                counts[f"count_incorrect_neg-{sample_type}"] += 1

    if not timing_only:
        counts_sorted = {k: counts[k] for k in sorted(counts)}

        overall_metrics = {
            "training_id": training_id,
            "training_type": training_type,
            "dataset": dataset_name,
            "weights_idx": weights_idx,
            "weights_file": class_weights_file,
            "random_seed": random_seed,
            "total_num_predictions": counts["count_correct"]
            + counts["count_incorrect"],
            **counts_sorted,
        }
        new_row = pd.DataFrame([overall_metrics])
        score_data = pd.concat((score_data, new_row), ignore_index=True)
        score_data.to_csv(out_file, index=False)

        # Store stats
        stats = {"time_elapsed": time.perf_counter() - start_time}
        tu.save_parameters(stats, "scores_classifier_stats", out_dir, txt_only=True)
    else:
        metrics = {
            "training_id": training_id,
            "training_type": training_type,
            "dataset": dataset_name,
            "weights_idx": weights_idx,
            "weights_file": class_weights_file,
            "random_seed": random_seed,
            "model": "classifier",
            **timing,
            "time_per_iteration": timing["time"] / timing["iterations"],
        }
        new_row = pd.DataFrame([metrics])
        score_data = pd.concat((score_data, new_row), ignore_index=True)
        score_data.to_csv(out_file, index=False)


if __name__ == "__main__":
    evaluate_class_run(
        PATHS,
        f"230406_162109_02_pcgan_--predicate_name-on_clutter_--encoder_id-230320_103300_02_pcenc_own_augment_1_0_hybrid-None",
        dataset_name="on_clutter",
        training_type="pcgan",
    )
