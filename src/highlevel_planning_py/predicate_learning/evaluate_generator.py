import os
import time
import subprocess
import logging
from datetime import datetime

import pandas as pd
from collections import Counter
import numpy as np
import torch
import torch_geometric as pyg
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec

import highlevel_planning_py.predicate_learning.training_utils as tu
from highlevel_planning_py.predicate_learning.dataset import (
    PredicateDataset,
    PredicateGraphDataset,
)
import highlevel_planning_py.predicate_learning.groundtruths_on as mcf_on
import highlevel_planning_py.predicate_learning.groundtruths_inside as mcf_inside
from highlevel_planning_py.predicate_learning.models_mlp import GeneratorMLP
from highlevel_planning_py.predicate_learning.models_gnn import GANgeneratorGNN
from highlevel_planning_py.predicate_learning.models_hybrid import (
    GANgeneratorHybridV1,
    GANgeneratorHybridV2,
)
from highlevel_planning_py.predicate_learning.models_pc_hybrid import (
    GANgeneratorHybridPCEncodings,
)
from highlevel_planning_py.predicate_learning.predicate_learning_server import SimServer
from highlevel_planning_py.predicate_learning.training_gan import (
    DatasetAdapterStandard,
    DatasetAdapterGraph,
)
from highlevel_planning_py.predicate_learning.dataset_pointcloud import (
    PredicateEncodingGraphDataset,
)
from highlevel_planning_py.predicate_learning.training_gan_pc_encoding import (
    DatasetAdapterPCEncoding,
)
from highlevel_planning_py.predicate_learning.models_sklearn import (
    DecisionTreeGenerator,
)
from highlevel_planning_py.predicate_learning.models_uniform_samplers import (
    UniformSampler,
)
from highlevel_planning_py.tools_pl import util
from highlevel_planning_py.tools_pl.logger_stdout import LoggerStdout


def evaluate_gen_run(
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
    """

    Args:
        paths:
        training_id:
        dataset_name:
        weights_idx:
        create_plots:
        random_seed:
        cuda:
        training_type:
        dataset_size:
        training_dir:

    Returns:

    """
    # Paths
    predicate_dir = os.path.join(paths["data_dir"], "predicates")
    if training_dir is None:
        training_dir = os.path.join(predicate_dir, "training", training_id)
    else:
        training_dir = os.path.join(training_dir, training_id)
    dataset_dir = os.path.join(predicate_dir, "data")

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

    pyg.seed_everything(random_seed)

    out_dir = os.path.join(training_dir, "post_eval")
    os.makedirs(out_dir, exist_ok=True)

    # Determine weights to use
    if training_type == "decisiontree":
        gen_weights_file = "classifier_final.pkl"
        gen_mdl_state = os.path.join(training_dir, gen_weights_file)
        gen_mdl_params = run_config
    elif training_type == "sampler":
        gen_weights_file = "dummy.txt"
        gen_mdl_params = run_config
        gen_mdl_state = None
    else:
        available_gen_weights = util.get_weights_list(training_dir, "gen")
        gen_weights_file = available_gen_weights[weights_idx]
        gen_mdl_params = tu.load_parameters(training_dir, "parameters_gen")
        if "out_indices_to_generate" not in gen_mdl_params.custom:
            gen_mdl_params.custom["out_indices_to_generate"] = [0, 1]
        gen_mdl_params.device = device
        state_file = os.path.join(training_dir, "models", gen_weights_file)
        gen_mdl_state = torch.load(state_file)

    if timing_only:
        out_file = os.path.join(out_dir, "timing_generator.csv")
        if os.path.isfile(out_file):
            logging.info("Already computed timing. Skipping.")
            return
        score_data = pd.DataFrame()
        extract_scores = True
    else:
        out_file = os.path.join(out_dir, "scores.csv")
        extract_scores = True
        if os.path.isfile(out_file):
            score_data = pd.read_csv(out_file)

            # Check if score was already computed
            mask = (
                (score_data["dataset"] == dataset_name)
                & (score_data["weights_file"] == gen_weights_file)
                & (score_data["random_seed"] == random_seed)
            )
            find_scores = score_data[mask]
            if len(find_scores) > 0:
                logging.info(
                    f"Already computed scores for {dataset_name}, "
                    f"{gen_weights_file} and seed {random_seed}. Skipping."
                )
                extract_scores = False
        else:
            score_data = pd.DataFrame()

    config_str = f"IMG-{dataset_name}-{gen_weights_file}-{random_seed}"
    img_out_dir = os.path.join(out_dir, config_str)
    if create_plots:
        os.makedirs(img_out_dir, exist_ok=True)

    # Load generator model
    logging.info(f"Using weights {gen_weights_file}")
    if training_type == "gan":
        if run_config.model_type == "mlp":
            model_gen = GeneratorMLP(gen_mdl_params)
            dataset_type = PredicateDataset
            dataset_adapter_type = DatasetAdapterStandard
            include_surrounding = False
        elif run_config.model_type == "hybrid":
            if run_config.model_version == "v1":
                model_gen = GANgeneratorHybridV1(gen_mdl_params)
            elif run_config.model_version == "v2":
                model_gen = GANgeneratorHybridV2(gen_mdl_params)
            else:
                raise NotImplementedError
            dataset_type = PredicateGraphDataset
            dataset_adapter_type = DatasetAdapterGraph
            include_surrounding = True
        else:
            model_gen = GANgeneratorGNN(gen_mdl_params)
            dataset_type = PredicateGraphDataset
            dataset_adapter_type = DatasetAdapterGraph
            include_surrounding = True
    elif training_type == "pcgan":
        if run_config.model_type == "hybrid":
            model_gen = GANgeneratorHybridPCEncodings(gen_mdl_params)
            dataset_type = PredicateEncodingGraphDataset
            dataset_adapter_type = DatasetAdapterPCEncoding
            include_surrounding = True
        else:
            raise NotImplementedError
    elif training_type == "decisiontree":
        model_gen = DecisionTreeGenerator(gen_mdl_params)
        dataset_type = (
            PredicateGraphDataset
            if run_config.feature_type == "manual"
            else PredicateEncodingGraphDataset
        )
        dataset_adapter_type = (
            DatasetAdapterGraph
            if run_config.feature_type == "manual"
            else DatasetAdapterPCEncoding
        )
        include_surrounding = True
    elif training_type == "sampler":
        model_gen = UniformSampler(gen_mdl_params, paths)
        dataset_type = (
            PredicateGraphDataset
            if run_config.feature_type == "manual"
            else PredicateEncodingGraphDataset
        )
        dataset_adapter_type = (
            DatasetAdapterGraph
            if run_config.feature_type == "manual"
            else DatasetAdapterPCEncoding
        )
        include_surrounding = True
    else:
        raise NotImplementedError
    model_gen.load_state_dict(gen_mdl_state)
    model_gen.to(device)
    model_gen.eval()

    # Set fixed random latent vectors
    camera_pos = (1.2, -1.7, 1.8)
    num_fixed = 9
    num_fixed_rows = 3
    assert num_fixed % num_fixed_rows == 0
    num_fixed_cols = int(num_fixed / num_fixed_rows)

    # Create dataset
    if training_type == "decisiontree" or training_type == "sampler":
        train_dataset_config = tu.load_parameters(
            training_dir, "parameters_data_class_train_val"
        )
        feature_version = run_config.feature_version
        fixed_latent = None
    else:
        train_dataset_config = tu.load_parameters(
            training_dir, "parameters_data_gen_train_val"
        )
        feature_version = (
            gen_mdl_params.feature_version
            if hasattr(gen_mdl_params, "feature_version")
            else run_config.dataset_feature_version
        )
        fixed_latent = model_gen.sample_noise(
            num_fixed, gen_mdl_params.custom["dim_in_noise"]
        ).to(device)

    dataset_gen_config = tu.DatasetConfig(
        device,
        dataset_dir,
        dataset_id,
        dataset_name,
        {"train": 1.0},
        target_model="gen",
        normalization_method=train_dataset_config.normalization_method,
        positive_only=False,
        include_surrounding=include_surrounding,
        feature_version=feature_version,
        encoder_id=train_dataset_config.encoder_id,
        dataset_size=dataset_size,
    )
    dataset = dataset_type(dataset_gen_config)
    dataset_adapter = dataset_adapter_type(device)

    # Simulator
    flags = util.parse_arguments(["--method", "direct"])
    sim_server = SimServer(
        flags,
        LoggerStdout(),
        paths,
        data_session_id=dataset_id,
        feature_version=feature_version,
        silent=True,
    )

    # Ground truth
    if "on_clutter" in dataset_name:
        classifier_params = [0.03, 0.04, 0.05]
        classifier_param_name = "gt_above_tol"
        # Legend reason: (not_too_low, not_too_high, within, aabb_overlap, cfree, result)
        reason_combinations = {
            key: 0
            for key in [
                "010010",
                "010110",
                "011110",
                "100010",
                "100110",
                "101110",
                "110010",
                "110110",
                "111100",
                "111111",
            ]
        }
        cl_names = ["on_aabb", "on_oabb", "on_cfree_oabb"]
        reason_classifier = "on_cfree_oabb"
        manual_classifiers = {
            cl_name: dict.fromkeys(classifier_params) for cl_name in cl_names
        }
        gt_results = {cl_name: dict.fromkeys(classifier_params) for cl_name in cl_names}
        gt_results_reasons = {reason_classifier: dict.fromkeys(classifier_params)}
        for param_value in classifier_params:
            manual_classifiers["on_aabb"][
                param_value
            ] = mcf_on.ManualClassifier_On_AABB(
                above_tol=param_value, device=device, feature_version=feature_version
            )
            manual_classifiers["on_oabb"][
                param_value
            ] = mcf_on.ManualClassifier_On_OABB(
                above_tol=param_value, device=device, feature_version=feature_version
            )
            manual_classifiers["on_cfree_oabb"][
                param_value
            ] = mcf_on.ManualClassifier_OnCfree_OABB(
                above_tol=param_value,
                device=device,
                feature_version=feature_version,
                always_check_collision=False,
            )
            for cl_name in cl_names:
                gt_results[cl_name][param_value] = {True: list(), False: list()}
            gt_results_reasons[reason_classifier][param_value] = {
                True: Counter(reason_combinations),
                False: Counter(reason_combinations),
            }
    elif "inside_drawer" in dataset_name:
        classifier_params = [0.001]
        classifier_param_name = "gt_pen_tol"
        # Legend reason: (not_too_high, not_too_low, not_outside_x, not_outside_y, cfree, result)
        reason_combinations = {
            key: 0
            for key in [
                "010010",
                "010110",
                "011010",
                "011110",
                "100010",
                "100110",
                "101010",
                "101110",
                "110010",
                "110110",
                "111010",
                "111100",
                "111111",
            ]
        }
        cl_names = ["inside_cfree_exact"]
        reason_classifier = "inside_cfree_exact"
        manual_classifiers = {
            cl_name: dict.fromkeys(classifier_params) for cl_name in cl_names
        }
        gt_results = {cl_name: dict.fromkeys(classifier_params) for cl_name in cl_names}
        gt_results_reasons = {reason_classifier: dict.fromkeys(classifier_params)}
        for param_value in classifier_params:
            if "inside_cfree_oabb" in manual_classifiers:
                manual_classifiers["inside_cfree_oabb"][
                    param_value
                ] = mcf_inside.ManualClassifier_Inside_OABB(
                    device, feature_version, pen_tolerance=param_value
                )
            manual_classifiers["inside_cfree_exact"][
                param_value
            ] = mcf_inside.ManualClassifier_Inside_OABB_TrueGeom(
                device,
                feature_version,
                pen_tolerance=param_value,
                paths=paths,
                data_session_id=dataset_id,
            )
            for cl_name in cl_names:
                gt_results[cl_name][param_value] = {True: list(), False: list()}
            gt_results_reasons[reason_classifier][param_value] = {
                True: Counter(reason_combinations),
                False: Counter(reason_combinations),
            }
    else:
        raise NotImplementedError

    # Loop through dataset, visualizing everything for every iteration
    len_dataset = min(len(dataset), dataset_size) if dataset_size >= 0 else len(dataset)
    num_plots = 10
    start_time = time.perf_counter()
    own_pid = os.getpid()
    max_memory = 0.0
    time_per_iteration = 0.0
    inference_time = 0.0
    i = 0
    timing = {"time": 0.0, "iterations": 0, "batch_size": num_fixed}
    for i in range(len_dataset):
        if (not create_plots or i >= num_plots) and not extract_scores:
            break

        features, features_others, label, demo_id, object_names = dataset.get_single(
            i, use_tensors=False
        )

        if training_type in ["decisiontree", "sampler"]:
            desired_labels = None
        elif run_config.pass_label_to_gen:
            desired_labels = torch.ones((num_fixed, 1)).to(device)
        else:
            desired_labels = None

        # Generate outputs
        input_scene = dataset_adapter.init_fixed_input(
            num_fixed, features, demo_id, dataset
        )
        time_start_inference = time.perf_counter()
        generated_scenes = model_gen.forward(
            input_scene, fixed_latent, desired_label=desired_labels
        )
        time_passed = time.perf_counter() - time_start_inference
        inference_time += time_passed
        timing["time"] += time_passed
        timing["iterations"] += 1

        if not timing_only:
            new_features_args = dataset_adapter.get_new_features_args(
                generated_scenes.detach(), dataset
            )

            demo_ids = [demo_id] * num_fixed

            # Append other features for ground truth computation
            if training_type in ["decisiontree", "sampler"]:
                generated_scenes_all = generated_scenes
            elif run_config.model_type == "mlp" and features_others.ndim > 1:
                if gen_mdl_params.custom["normalize_output"]:
                    raise NotImplementedError(
                        "Normalization would set generated into frame that is different from the frame of other features."
                    )
                features_others_expanded = dataset_adapter.init_fixed_input(
                    num_fixed, features_others, demo_id, dataset
                )
                generated_scenes_all = torch.cat(
                    (generated_scenes, features_others_expanded), dim=1
                )
            else:
                generated_scenes_all = generated_scenes

            # Compute ground truth
            with torch.no_grad():
                for param_value in classifier_params:
                    for cl_name, classifier in manual_classifiers.items():
                        gt_performance, gt_reason = dataset_adapter.evaluate_gt_classifier(
                            generated_scenes_all,
                            classifier[param_value],
                            dataset,
                            get_reason=True,
                            demo_ids=demo_ids,
                        )
                        gt_results[cl_name][param_value][bool(label)].extend(
                            gt_performance.detach().tolist()
                        )
                        if cl_name in gt_results_reasons:
                            gt_results_reasons[cl_name][param_value][
                                bool(label)
                            ].update(gt_reason)

            # Plot everything
            if create_plots and i < num_plots:
                filename = os.path.join(img_out_dir, f"{demo_id}.png")
                if os.path.isfile(filename):
                    continue

                sim_server.world.add_ground_plane()
                sim_server.scene.objects = dict()
                sim_server.pfm.restore_demonstration_outside(dataset_name, demo_id)
                all_features = (
                    np.concatenate((features, features_others))
                    if len(features_others) > 0
                    else features
                )
                all_names = object_names["args"] + object_names["others"]
                sim_server.visualize_features(
                    all_features,
                    all_names,
                    show_ground=False,
                    remove_nonmentioned_objects=False,
                )

                # Capture image of input
                img_in = sim_server.world.capture_image(camera_pos=camera_pos)

                fig = plt.figure(constrained_layout=True, figsize=(15, 10))
                gs = GridSpec(3, 4, figure=fig)
                ax_demo = fig.add_subplot(gs[2, 0])
                ax_demo.imshow(img_in)
                ax_demo.axis("off")
                ax_demo.set_title("Input")

                axs_gen = dict()
                for idx in range(len(new_features_args)):
                    all_features = (
                        np.concatenate((new_features_args[idx], features_others))
                        if len(features_others) > 0
                        else new_features_args[idx]
                    )
                    all_names = object_names["args"] + object_names["others"]
                    sim_server.visualize_features(
                        all_features,
                        all_names,
                        show_ground=False,
                        remove_nonmentioned_objects=False,
                    )
                    img = sim_server.world.capture_image(camera_pos=camera_pos)
                    plot_row = int(np.floor(idx / num_fixed_cols))
                    plot_col = idx % num_fixed_cols + 1
                    axs_gen[idx] = fig.add_subplot(gs[plot_row, plot_col])
                    axs_gen[idx].imshow(img)
                    axs_gen[idx].axis("off")
                    axs_gen[idx].set_title(
                        f"gt AABB: {gt_results['on_aabb'][classifier_params[0]][bool(label)][- len(new_features_args) + idx]} "
                        f"gt OABB: {gt_results['on_oabb'][classifier_params[0]][bool(label)][- len(new_features_args) + idx]}"
                    )

                metrics = {"Demo ID": demo_id, "Label": bool(label)}
                ax_metrics = fig.add_subplot(gs[:2, 0])
                for i_metric, metric in enumerate(metrics.items()):
                    ax_metrics.text(
                        0.1, 0.8 - i_metric * 0.1, f"{metric[0]}: {metric[1]}"
                    )
                ax_metrics.axis("off")

                plt.savefig(filename, dpi=250)
                plt.close(fig)
            else:
                sim_server.close()

        # Status
        if (i + 1) % 500 == 0:
            time_per_iteration = (time.perf_counter() - start_time) / (i + 1)
            logging.info(
                f"Processed {i + 1}/{len_dataset}. Time per iteration: {time_per_iteration:.4f}. "
                f"Time elapsed: {time.perf_counter() - start_time:.2f}. "
                f"Time left: {time_per_iteration * (len_dataset - i - 1):.2f}."
            )
            try:
                res = subprocess.run(
                    ["ps", "-p", str(own_pid), "-o", "rss,pmem"],
                    capture_output=True,
                    text=True,
                ).stdout
                res = res.split("\n")[1]
                res = res.split(" ")
                res = [item for item in res if item != ""]
                mem_gb_current = float(res[0]) / 1000000.0
                mem_percent_current = float(res[1])
            except IndexError:
                mem_gb_current = -1
                mem_percent_current = -1
            if mem_gb_current > max_memory:
                max_memory = mem_gb_current
            logging.info(
                f"Memory usage: {mem_gb_current:.2f} GB ({mem_percent_current:.2f} %)."
            )

    sim_server.close()

    if extract_scores:
        if not timing_only:
            for param_value in classifier_params:
                reasons_pos = {
                    f"reason_pos_{rlabel}": count
                    for rlabel, count in sorted(
                        gt_results_reasons[reason_classifier][param_value][
                            True
                        ].items(),
                        reverse=True,
                    )
                }
                reasons_neg = {
                    f"reason_neg_{rlabel}": count
                    for rlabel, count in sorted(
                        gt_results_reasons[reason_classifier][param_value][
                            False
                        ].items(),
                        reverse=True,
                    )
                }

                avg_groundtruths = dict()
                for cl_name in cl_names:
                    avg_groundtruths[f"avg_gt_{cl_name}"] = np.mean(
                        gt_results[cl_name][param_value][True]
                        + gt_results[cl_name][param_value][False]
                    )
                    avg_groundtruths[f"avg_gt_{cl_name}_pos"] = np.mean(
                        gt_results[cl_name][param_value][True]
                    )
                    avg_groundtruths[f"avg_gt_{cl_name}_neg"] = np.mean(
                        gt_results[cl_name][param_value][False]
                    )
                overall_metrics = {
                    "training_id": training_id,
                    "training_type": training_type,
                    "dataset": dataset_name,
                    "weights_idx": weights_idx,
                    "weights_file": gen_weights_file,
                    "random_seed": random_seed,
                    classifier_param_name: param_value,
                    "total_num_predictions": len(
                        list(gt_results.values())[0][param_value][True]
                        + list(gt_results.values())[0][param_value][False]
                    ),
                    **avg_groundtruths,
                    **reasons_pos,
                    **reasons_neg,
                }
                new_row = pd.DataFrame([overall_metrics])
                score_data = pd.concat((score_data, new_row), ignore_index=True)
            score_data.to_csv(out_file, index=False)

            # Store stats
            stats = {
                "dataset_name": dataset_name,
                "weights_idx": weights_idx,
                "random_seed": random_seed,
                "dataset_size": dataset_size,
                "eval_time_per_iteration": time_per_iteration,
                "eval_time_elapsed": time.perf_counter() - start_time,
                "max_memory": max_memory,
                "inference_time": inference_time,
                "iterations": i + 1,
            }
            time_now = datetime.now()
            time_string = time_now.strftime("%y%m%d_%H%M%S")
            tu.save_parameters(stats, f"{time_string}_scores_stats", out_dir)
        else:
            metrics = {
                "training_id": training_id,
                "training_type": training_type,
                "dataset": dataset_name,
                "weights_idx": weights_idx,
                "weights_file": gen_weights_file,
                "random_seed": random_seed,
                "model": "generator",
                **timing,
                "time_per_iteration": timing["time"] / timing["iterations"],
            }
            new_row = pd.DataFrame([metrics])
            score_data = pd.concat((score_data, new_row), ignore_index=True)
            score_data.to_csv(out_file, index=False)

    del sim_server
    del model_gen
    del dataset
    for _, cl_dict in manual_classifiers.items():
        for _, cl in cl_dict.items():
            cl.close()
    del manual_classifiers

    logging.info("Success!")
