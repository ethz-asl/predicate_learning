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
import sys
import time
import pickle
from copy import deepcopy
from itertools import product

from torch_geometric import seed_everything
import argparse
import logging
import sklearn
from sklearn import svm, tree
from highlevel_planning_py.predicate_learning.dataset import (
    PredicateGraphDataset,
    build_sklearn_dataset,
)
from highlevel_planning_py.predicate_learning import training_utils as tu

import torch
from torch import nn

from highlevel_planning_py.predicate_learning.dataset_pointcloud import (
    PredicateEncodingGraphDataset,
)
from highlevel_planning_py.predicate_learning.evaluate_classifier import (
    evaluate_class_run,
)
from highlevel_planning_py.predicate_learning.evaluate_generator import evaluate_gen_run
from highlevel_planning_py.tools_pl.path import get_path_dict


def parse_arguments():
    parser = argparse.ArgumentParser()
    tu.parse_general_args(parser)
    parser.add_argument("--classifier_model", type=str, choices=["svm", "dt"])
    parser.add_argument("--feature_version", type=str, default="v1")
    parser.add_argument("--data_normalization_class", type=str, default="none")
    parser.add_argument(
        "--criterion", type=str, default="gini", choices=["gini", "entropy", "log_loss"]
    )
    parser.add_argument(
        "--splitter", type=str, default="best", choices=["best", "random"]
    )
    parser.add_argument("--max_depth", type=int, default=-1)
    parser.add_argument("--feature_type", type=str, choices=["manual", "pcenc"])
    parser.add_argument("--encoder_id", type=str, default="")
    args = parser.parse_args()
    return args


def compute_metrics(pred, x, y, clf, label: str):
    loss_fn = nn.BCELoss()
    loss_t = loss_fn(torch.from_numpy(pred), torch.from_numpy(y).double()).item()
    loss_s = sklearn.metrics.log_loss(y, pred)
    acc = clf.score(x, y)
    print(f"{label} loss: torch {loss_t}, sklearn {loss_s}")
    print(f"{label} acc: {acc}")
    return {"loss_torch": loss_t, "loss_sklearn": loss_s, "accuracy": acc}


def main(args):
    paths = get_path_dict(args.paths)
    predicate_dir = os.path.join(paths["data_dir"], "predicates")
    device = "cpu" if not args.gpu else "cuda"

    out_dir = tu.setup_out_dir(
        predicate_dir,
        resume_training=False,
        resume_timestring=args.id_string,
        screen_output=args.screen_output,
        training_type="decisiontree",
    )
    logging.info(f"Command: {' '.join(sys.argv)}")

    tu.save_parameters(vars(args), "parameters_training", out_dir, parameters_obj=args)

    seed_everything(args.random_seed)

    # Create dataset
    dataset_dir = os.path.join(paths["data_dir"], "predicates", "data")
    dataset_config = tu.DatasetConfig(
        device,
        dataset_dir,
        args.dataset_id,
        args.predicate_name,
        splits={},
        target_model="class",
        normalization_method=args.data_normalization_class,
        positive_only=False,
        include_surrounding=False,
        feature_version=args.feature_version,
        encoder_id=args.encoder_id,
    )
    test_dataset_config = deepcopy(dataset_config)
    test_dataset_config.predicate_name += "_test"
    if args.feature_type == "manual":
        dataset = PredicateGraphDataset(dataset_config)
        test_dataset = PredicateGraphDataset(test_dataset_config)
    elif args.feature_type == "pcenc":
        dataset = PredicateEncodingGraphDataset(dataset_config)
        test_dataset = PredicateEncodingGraphDataset(test_dataset_config)
    else:
        raise ValueError(f"Unknown feature type {args.feature_type}")

    tu.save_parameters(
        dataset_config.to_dict(),
        "parameters_data_class_train_val",
        out_dir,
        parameters_obj=dataset_config,
    )
    tu.save_parameters(
        test_dataset_config.to_dict(),
        "parameters_data_class_test",
        out_dir,
        parameters_obj=test_dataset_config,
    )

    val_split = 0.2
    shuffle_dataset = True

    # Training and validation split
    dataset_size = len(dataset) if args.dataset_size == -1 else args.dataset_size
    train_indices, val_indices, _ = tu.train_test_split(
        dataset_size, val_split, shuffle_dataset=shuffle_dataset
    )

    # Get data
    x, y = build_sklearn_dataset(dataset)
    x_test, y_test = build_sklearn_dataset(test_dataset)
    x_train, y_train = x[train_indices, :], y[train_indices]
    x_val, y_val = x[val_indices, :], y[val_indices]
    y_train = y_train.astype(int)
    y_val = y_val.astype(int)
    y_test = y_test.astype(int)

    stats = dict()

    # Setup model
    if args.classifier_model == "svm":
        clf = svm.SVC(probability=True)
    elif args.classifier_model == "dt":
        clf = tree.DecisionTreeClassifier(
            criterion=args.criterion,
            splitter=args.splitter,
            max_depth=args.max_depth if args.max_depth > 0 else None,
        )
    else:
        raise ValueError(f"Unknown classifier model {args.classifier_model}")

    # Train
    start_time = time.perf_counter()
    clf.fit(x_train, y_train)
    stats["training_time"] = time.perf_counter() - start_time

    if args.classifier_model == "dt":
        stats["tree_depth"] = clf.get_depth()
        stats["tree_n_leaves"] = clf.get_n_leaves()

    with open(os.path.join(out_dir, "classifier_final.pkl"), "wb") as f:
        pickle.dump(clf, f)

    # Show performance
    pred_train = clf.predict_proba(x_train)
    pred_train = pred_train[:, 1]
    pred_val = clf.predict_proba(x_val)
    pred_val = pred_val[:, 1]
    pred_test = clf.predict_proba(x_test)
    pred_test = pred_test[:, 1]

    metrics_train = compute_metrics(pred_train, x_train, y_train, clf, "train")
    metrics_val = compute_metrics(pred_val, x_val, y_val, clf, "val")
    metrics_test = compute_metrics(pred_test, x_test, y_test, clf, "test")
    stats["metrics_train"] = metrics_train
    stats["metrics_val"] = metrics_val
    stats["metrics_test"] = metrics_test
    tu.save_parameters(stats, "stats", out_dir)

    logging.info("Finished training successfully.")

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
                    training_type="decisiontree",
                )
            if args.evaluate_class:
                evaluate_class_run(
                    paths,
                    out_dir.split("/")[-1],
                    config[0],
                    config[1],
                    random_seed=config[2],
                    training_type="decisiontree",
                )
        logging.info("Evaluation done")


if __name__ == "__main__":
    arguments = parse_arguments()
    main(arguments)
