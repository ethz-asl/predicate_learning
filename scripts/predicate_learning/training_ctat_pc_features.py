#!/usr/bin/env python

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

# "ctat" stands for "classifier training, adversarial training"

import os
import sys
from itertools import product
import argparse
import logging
from copy import deepcopy
import time
import atexit
from collections import defaultdict
import torch_geometric as pyg
from torch import cuda

from highlevel_planning_py.predicate_learning.models_hybrid import (
    GANdiscriminatorHybrid,
)
from highlevel_planning_py.predicate_learning.models_mlp import DiscriminatorMLP
from highlevel_planning_py.predicate_learning.models_pc_hybrid import (
    GANgeneratorHybridPCEncodings,
    GeneratorMLPPCEncodings,
)
from highlevel_planning_py.predicate_learning.groundtruths_on import (
    ManualClassifier_OnCfree_OABB,
)
from highlevel_planning_py.predicate_learning.groundtruths_inside import (
    ManualClassifier_Inside_OABB,
)
from highlevel_planning_py.predicate_learning import training_utils as tu
from highlevel_planning_py.predicate_learning.training_gan_pc_encoding import (
    TrainingSequenceGANctatPCEnc,
)
from highlevel_planning_py.predicate_learning.dataset_pointcloud import (
    PredicateEncodingGraphDataset,
)
from highlevel_planning_py.predicate_learning.evaluate_generator import evaluate_gen_run
from highlevel_planning_py.predicate_learning.evaluate_classifier import (
    evaluate_class_run,
)
from highlevel_planning_py.tools_pl import util
from highlevel_planning_py.tools_pl.path import get_path_dict


def parse_arguments():
    parser = argparse.ArgumentParser()
    tu.parse_general_args(parser)
    tu.parse_general_ctat_args(parser)

    parser.add_argument(
        "--encoder_id",
        action="store",
        default="220914_180008_12_--predicate_name-on_clutter_--encoder_type-conv_--decoder_type-fc_--dataset_normalization-all_data_max",
    )
    parser.add_argument(
        "--inflate_features",
        type=util.string_to_list,
        default=[0, 0],
        help="start and end indices of features to inflate",
    )
    parser.add_argument(
        "--inflate_to", type=int, default=10, help="length to inflate to"
    )

    subparsers = parser.add_subparsers()
    parser_mlp = subparsers.add_parser("mlp")
    tu.parse_arguments_mlp(parser_mlp)
    # parser_hybrid = subparsers.add_parser("gnn")
    # parse_arguments_hybrid(parser_hybrid, "gnn")
    parser_hybrid = subparsers.add_parser("hybrid")
    tu.parse_arguments_hybrid(parser_hybrid, "hybrid")
    args = parser.parse_args()
    return args


def main(args):
    paths = get_path_dict(args.paths)
    predicate_dir = os.path.join(paths["data_dir"], "predicates")

    if args.gpu:
        assert cuda.is_available(), "CUDA not available"

    resume_training = args.resume
    resume_timestring = args.id_string
    resume_additional_it = {"ct": args.num_class_it, "at": args.num_adversarial_it}

    out_dir = tu.setup_out_dir(
        predicate_dir,
        resume_training,
        resume_timestring,
        screen_output=args.screen_output,
        training_type="pcgan",
    )
    logging.info(f"Command: {' '.join(sys.argv)}")

    if args.dataset_id == "220509-100351_demonstrations_features":
        fixed_scene_id = "220509-100627-741247"
    elif args.dataset_id == "220608-190256_demonstrations_features":
        fixed_scene_id = "220608-190455-592478"
    else:
        fixed_scene_id = ""

    if args.save_period == "normal":
        save_periods = {
            "save_model_period": 7000,
            "save_img_period": 3000,
            "train_log_period": 100,
        }
    elif args.save_period == "long":
        save_periods = {
            "save_model_period": 20000,
            "save_img_period": 10000,
            "train_log_period": 100,
        }
    else:
        raise ValueError

    # Parameters
    run_config = tu.TrainingConfig(
        device="cpu" if not args.gpu else "cuda",
        random_seed=args.random_seed,
        model_type=args.model_type,
        model_version=args.model_version,
        predicate_name=args.predicate_name,
        dataset_id=args.dataset_id,
        dataset_feature_version=args.feature_version,
        save_img_fixed_scene_id=fixed_scene_id,
        dataset_size=args.dataset_size,
        loss_aggregation=args.loss_aggregation,
        batch_size=args.batch_size,
        training_style="schedule",
        training_schedule={"ct": args.num_class_it, "at": args.num_adversarial_it},
        disc_iters_per_gen_iter=5,
        eval_period=250,
        dim_noise=4,
        optimizer="rmsprop",
        learning_rate=args.learning_rate,
        use_lr_scheduler=False,
        pass_label_to_disc=args.disc_pass_label,
        pass_label_to_gen=args.gen_pass_label,
        loss="wasserstein",
        clipper_type="none",
        clipper_radius=1.0,
        gradient_penalty=True,
        gradient_penalty_lambda=10,
        gen_loss_components=args.gen_loss_components,
        **save_periods,
    )
    pyg.seed_everything(run_config.random_seed)
    stats, run_config = tu.training_admin_beginning(
        run_config, out_dir, resume_training, resume_additional_it
    )

    # Create datasets
    dataset_type = PredicateEncodingGraphDataset
    include_surrounding = (
        args.include_surrounding if hasattr(args, "include_surrounding") else False
    )
    dataset_dir = os.path.join(predicate_dir, "data")
    dataset_configs = list()
    dataset_types = list()
    dataset_class_config = tu.DatasetConfig(
        run_config.device,
        dataset_dir,
        run_config.dataset_id,
        args.predicate_name,
        {"train": 0.9, "val": 0.1},
        target_model="class",
        normalization_method=args.data_normalization_class,
        positive_only=False,
        include_surrounding=include_surrounding,
        dataset_size=run_config.dataset_size,
        encoder_id=args.encoder_id,
    )
    dataset_configs.append(dataset_class_config)
    dataset_types.append(dataset_type)
    dataset_class_test_config = deepcopy(dataset_class_config)
    dataset_class_test_config.predicate_name = args.predicate_name + "_test"
    dataset_class_test_config.splits = {"test": 1.0}
    dataset_class_test_config.dataset_size = -1
    dataset_configs.append(dataset_class_test_config)
    dataset_types.append(dataset_type)
    dataset_disc_config = tu.DatasetConfig(
        run_config.device,
        dataset_dir,
        run_config.dataset_id,
        args.predicate_name,
        {"train": 0.9, "val": 0.1},
        target_model="disc",
        normalization_method=args.data_normalization_disc_gen,
        positive_only=True,
        include_surrounding=include_surrounding,
        dataset_size=run_config.dataset_size,
        encoder_id=args.encoder_id,
    )
    dataset_configs.append(dataset_disc_config)
    dataset_types.append(dataset_type)
    dataset_disc_test_config = deepcopy(dataset_disc_config)
    dataset_disc_test_config.predicate_name = args.predicate_name + "_test"
    dataset_disc_test_config.splits = {"test": 1.0}
    dataset_disc_test_config.dataset_size = -1
    dataset_configs.append(dataset_disc_test_config)
    dataset_types.append(dataset_type)
    dataset_gen_config = tu.DatasetConfig(
        run_config.device,
        dataset_dir,
        run_config.dataset_id,
        args.predicate_name,
        {"train": 0.9, "val": 0.1},
        target_model="gen",
        normalization_method=args.data_normalization_disc_gen,
        positive_only=False,
        include_surrounding=include_surrounding,
        dataset_size=run_config.dataset_size,
        encoder_id=args.encoder_id,
    )
    dataset_configs.append(dataset_gen_config)
    dataset_types.append(dataset_type)
    dataset_gen_test_config = deepcopy(dataset_gen_config)
    dataset_gen_test_config.predicate_name = args.predicate_name + "_test"
    dataset_gen_test_config.splits = {"test": 1.0}
    dataset_gen_test_config.dataset_size = -1
    dataset_configs.append(dataset_gen_test_config)
    dataset_types.append(dataset_type)

    loaders = defaultdict(dict)
    datasets = defaultdict(dict)
    dimensions = dict()
    tu.setup_data(
        loaders,
        datasets,
        dimensions,
        resume_training,
        out_dir,
        run_config.batch_size,
        dataset_configs,
        dataset_types,
    )

    # Model parameters
    out_indices_to_generate = [1]
    gen_custom = {
        "quat_idx_out": 3,
        "dim_in_noise": run_config.dim_noise,
        "out_indices_to_generate": out_indices_to_generate,
        # "normalize_output": args.gen_normalize_output,
        # Should be normalized according to what classifier expects
        # "normalization_method": args.data_normalization_class,
    }
    # if run_config.dataset_feature_version == "v1":
    #     disc_class_feature_selector = tuple(range(dimensions["disc"]))
    #     gen_custom["idx_in_oabb"] = 13
    #     gen_custom["in_feature_selector"] = tuple(range(dimensions["gen"]))
    # elif run_config.dataset_feature_version == "v2":
    #     disc_class_feature_selector = tuple(range(19))
    #     gen_custom["idx_in_oabb"] = 19
    #     gen_custom["in_feature_selector"] = tuple(range(19))
    # elif run_config.dataset_feature_version == "v3":
    #     disc_class_feature_selector = tuple(range(16))
    #     gen_custom["idx_in_oabb"] = 16
    #     gen_custom["in_feature_selector"] = tuple(range(16))
    # else:
    #     raise ValueError

    if run_config.model_type == "mlp":
        if args.inflate_features[1] > args.inflate_features[0]:
            raise NotImplementedError
        class_params = tu.MLPNetworkConfig(
            dim_in=2 * dimensions["class"],
            dim_out=1,
            layers=args.class_layers,
            device=run_config.device,
            feature_version=run_config.dataset_feature_version,
            initialize_with=args.init_class_with,
            activation_type=args.class_activation,
        )
        class_model_type = DiscriminatorMLP
        disc_params = tu.MLPNetworkConfig(
            dim_in=2 * dimensions["disc"],
            dim_out=1,
            layers=args.disc_layers,
            device=run_config.device,
            feature_version=run_config.dataset_feature_version,
            activation_type=args.disc_activation,
        )
        disc_model_type = DiscriminatorMLP
        gen_params = tu.MLPNetworkConfig(
            dim_in=run_config.dim_noise + 2 * dimensions["gen"],
            dim_out=7 * len(out_indices_to_generate),
            layers=args.gen_layers,
            custom={**gen_custom, "out_complete_scene": True},
            device=run_config.device,
            feature_version=run_config.dataset_feature_version,
            activation_type=args.gen_activation,
        )
        gen_model_type = GeneratorMLPPCEncodings
    elif run_config.model_type == "gnn":
        raise NotImplementedError
    elif run_config.model_type == "hybrid":
        if args.class_encoder_type in ["own", "gat"]:
            if args.inflate_features[1] > args.inflate_features[0]:
                raise NotImplementedError
            class_config_encoder = tu.GNNNetworkConfig(
                dim_in=dimensions["disc"],
                dim_out=args.scene_encoding_dim,
                layers=args.class_encoder_layers,
                dim_inner=args.class_gnn_dim_inner,
                graph_pooling="add",
                post_processing=False,
                device=run_config.device,
                custom={"jk": "last"},
            )
        elif args.class_encoder_type == "mlp":
            class_config_encoder = tu.MLPNetworkConfig(
                dim_out=args.scene_encoding_dim,
                layers=args.class_encoder_layers,
                custom={"graph_pooling": "mean"},
                device=run_config.device,
            )
        else:
            raise ValueError
        class_params = tu.HybridNetworkConfig(
            device=run_config.device,
            num_argument_objects=2,
            num_features=dimensions["disc"],
            config_encoder=class_config_encoder,
            config_main_net=tu.MLPNetworkConfig(
                dim_out=1,
                layers=args.class_main_net_layers,
                feature_version=run_config.dataset_feature_version,
                device=run_config.device,
            ),
            encoder_type=args.class_encoder_type,
            initialize_with=args.init_class_with,
            inflate_features=tuple(args.inflate_features),
            inflate_to=args.inflate_to,
        )
        class_params.complete()
        class_params.verify()
        class_model_type = GANdiscriminatorHybrid

        if args.disc_encoder_type in ["own", "gat"]:
            if args.inflate_features[1] > args.inflate_features[0]:
                raise NotImplementedError
            disc_config_encoder = tu.GNNNetworkConfig(
                dim_in=dimensions["disc"],
                dim_out=args.scene_encoding_dim,
                layers=args.disc_encoder_layers,
                dim_inner=args.disc_gnn_dim_inner,
                graph_pooling="add",
                post_processing=False,
                device=run_config.device,
            )
        elif args.disc_encoder_type == "mlp":
            disc_config_encoder = tu.MLPNetworkConfig(
                dim_out=args.scene_encoding_dim,
                layers=args.disc_encoder_layers,
                custom={"graph_pooling": "mean"},
                device=run_config.device,
            )
        else:
            raise ValueError
        disc_params = tu.HybridNetworkConfig(
            device=run_config.device,
            num_argument_objects=2,
            num_features=dimensions["disc"],
            config_encoder=disc_config_encoder,
            config_main_net=tu.MLPNetworkConfig(
                dim_out=1,
                layers=args.disc_main_net_layers,
                feature_version=run_config.dataset_feature_version,
                device=run_config.device,
            ),
            encoder_type=args.disc_encoder_type,
            inflate_features=tuple(args.inflate_features),
            inflate_to=args.inflate_to,
        )
        disc_params.complete()
        disc_params.verify()
        disc_model_type = GANdiscriminatorHybrid

        if args.gen_encoder_type in ["gat", "own"]:
            if args.inflate_features[1] > args.inflate_features[0]:
                raise NotImplementedError
            gen_encoder_config = tu.GNNNetworkConfig(
                dim_in=dimensions["gen"],
                dim_out=args.scene_encoding_dim,
                custom=gen_custom,
                layers=args.gen_encoder_layers,
                dim_inner=args.gen_gnn_dim_inner,
                graph_pooling="add",
                post_processing=False,
                device=run_config.device,
            )
        elif args.gen_encoder_type == "mlp":
            gen_encoder_config = tu.MLPNetworkConfig(
                dim_out=args.scene_encoding_dim,
                layers=args.gen_encoder_layers,
                custom={"graph_pooling": "mean"},
                device=run_config.device,
            )
        else:
            raise ValueError
        gen_params = tu.HybridNetworkConfig(
            num_argument_objects=2,
            num_features=dimensions["gen"],
            custom=gen_custom,
            feature_version=run_config.dataset_feature_version,
            config_encoder=gen_encoder_config,
            config_main_net=tu.MLPNetworkConfig(
                dim_out=7,
                layers=args.gen_main_net_layers,
                custom={
                    "quat_idx_out": gen_custom["quat_idx_out"],
                    "out_indices_to_generate": gen_custom["out_indices_to_generate"],
                    "normalize_output": False,
                },
                device=run_config.device,
            ),
            encoder_type=args.gen_encoder_type,
            device=run_config.device,
            inflate_features=tuple(args.inflate_features),
            inflate_to=args.inflate_to,
        )
        gen_params.complete()
        gen_params.verify()
        gen_params.config_main_net.dim_in += run_config.dim_noise
        gen_model_type = GANgeneratorHybridPCEncodings
    else:
        raise ValueError("Invalid model type")

    # Models
    model_class, n_param_class, latest_iteration_class = tu.create_model(
        class_model_type,
        class_params,
        resume_training,
        run_config.device,
        out_dir,
        predicate_dir,
        "class",
    )
    stats = util.dict_set_without_overwrite(stats, "n_param_class", n_param_class)
    model_disc, n_param_disc, latest_iteration_disc = tu.create_model(
        disc_model_type,
        disc_params,
        resume_training,
        run_config.device,
        out_dir,
        predicate_dir,
        "disc",
    )
    stats = util.dict_set_without_overwrite(stats, "n_param_disc", n_param_disc)
    model_gen, n_param_gen, latest_iteration_gen = tu.create_model(
        gen_model_type,
        gen_params,
        resume_training,
        run_config.device,
        out_dir,
        predicate_dir,
        "gen",
    )
    stats = util.dict_set_without_overwrite(stats, "n_param_gen", n_param_gen)

    if (latest_iteration_class != latest_iteration_gen) or (
        latest_iteration_class != latest_iteration_disc
    ):
        raise ValueError("Latest iterations of models do not match")

    if resume_training and stats["total_iterations"] < latest_iteration_class:
        stats["total_iterations"] = latest_iteration_class

    model_class_container = tu.ModelSplitter(model_class, 0)
    model_disc_container = tu.ModelSplitter(model_disc, 0)

    if args.run_gt_classifier:
        if args.predicate_name == "on_clutter":
            gt_classifier = ManualClassifier_OnCfree_OABB(
                above_tol=0.03,
                device=run_config.device,
                feature_version=run_config.dataset_feature_version,
            )
        elif args.predicate_name == "inside_drawer":
            gt_classifier = ManualClassifier_Inside_OABB(
                run_config.device, run_config.dataset_feature_version, pen_tolerance=0.0
            )
        else:
            raise NotImplementedError
    else:
        gt_classifier = None

    # Train
    ts = TrainingSequenceGANctatPCEnc(
        run_config,
        model_class_container,
        model_disc_container,
        model_gen,
        run_config.device,
        out_dir,
        loaders,
        paths,
        resume=resume_training,
        debug=False,
        dataset=datasets["gen"],
        ground_truth_classifier=gt_classifier,
    )

    logging.info("Start training")
    start_time = time.time()
    if not args.explicit_exit_handler:
        atexit.register(tu.exit_handler, start_time, stats, out_dir, ts.loggers)

    tu.run_training(ts, run_config, stats, out_dir)
    logging.info("Finished training.")

    # Evaluate
    if args.evaluate_gen or args.evaluate_class:
        logging.info("Evaluating run")
        dataset_names = [args.predicate_name + "_test"]
        weight_indices = [-1]
        random_seeds = [12]
        configs = tuple(product(dataset_names, weight_indices, random_seeds))
        for config in configs:
            if args.evaluate_gen:
                evaluate_gen_run(
                    paths,
                    out_dir.split("/")[-1],
                    config[0],
                    config[1],
                    random_seed=config[2],
                    training_type="pcgan",
                )
            if args.evaluate_class:
                evaluate_class_run(
                    paths,
                    out_dir.split("/")[-1],
                    config[0],
                    config[1],
                    random_seed=config[2],
                    training_type="pcgan",
                )
        logging.info("Evaluation done")

    if args.explicit_exit_handler:
        tu.exit_handler(start_time, stats, out_dir, ts.loggers)


if __name__ == "__main__":
    arguments = parse_arguments()
    main(arguments)
