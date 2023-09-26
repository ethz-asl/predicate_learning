#!/usr/bin/env python3

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

import logging
import os
import argparse
from itertools import product
from highlevel_planning_py.predicate_learning.evaluate_generator import evaluate_gen_run
from highlevel_planning_py.predicate_learning.evaluate_classifier import (
    evaluate_class_run,
)
from highlevel_planning_py.tools_pl.path import get_path_dict


def parse_arguments(args_to_parse=None):
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--paths",
        type=str,
        choices=["local", "euler"],
        default="local",
        help="Paths to use",
    )
    parser.add_argument(
        "--model", type=str, choices=["gen", "class"], help="Model type"
    )
    parser.add_argument("--cuda", action="store_true", help="Use GPU")
    parser.add_argument("--filter_str", type=str, default="", help="Runs to evaluate")
    parser.add_argument("--additional_filter_str", type=str, default=None)
    parser.add_argument(
        "--training_type", type=str, choices=["gan", "pcgan", "decisiontree", "sampler"]
    )
    parser.add_argument("--random_seeds", nargs="*", type=int, default=[12])
    parser.add_argument("--create_plots", action="store_true")
    parser.add_argument("--dataset_names", nargs="*", default=None)
    parser.add_argument("--autodetect_dataset_names", action="store_true")
    parser.add_argument(
        "--eval_on_split",
        type=str,
        default="test",
        choices=["train", "test", "both"],
        help="Which split to evaluate on",
    )
    parser.add_argument("--dataset_size", type=int, default=-1)
    parser.add_argument("--weight_indices", nargs="*", type=int, default=[-1])
    parser.add_argument("--training_dir", type=str, default=None)
    parser.add_argument(
        "--timing_only",
        action="store_true",
        help="Do not evaluate performance, but only inference time.",
    )
    parser.add_argument("--dry_run", action="store_true")
    if args_to_parse is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(args_to_parse)
    return args


def evaluate_multiple(args):
    paths = get_path_dict(args.paths)

    if args.model == "gen":
        eval_func = evaluate_gen_run
    elif args.model == "class":
        eval_func = evaluate_class_run
    else:
        raise ValueError(f"Unknown model type {args.model}")

    if args.training_dir is None:
        predicate_dir = os.path.join(paths["data_dir"], "predicates")
        training_dir = os.path.join(predicate_dir, "training")
    else:
        training_dir = args.training_dir
    logging.info(f"Training dir: {training_dir}")
    available_trainings = os.listdir(training_dir)
    available_trainings = [
        item
        for item in available_trainings
        if args.filter_str in item
        and os.path.isdir(os.path.join(training_dir, item))
        and args.training_type in item
    ]
    if args.additional_filter_str is not None:
        available_trainings = [
            item for item in available_trainings if args.additional_filter_str in item
        ]
    available_trainings.sort()
    logging.info("Evaluating the following trainings:")
    for tr in available_trainings:
        logging.info(tr)

    if args.dry_run:
        return

    if args.autodetect_dataset_names:
        if args.dataset_names is not None:
            logging.warning(
                "dataset_names is set, but autodetect_dataset_names is True. Ignoring dataset_names."
            )
        if args.eval_on_split == "train":
            dataset_names = ["detect"]
        elif args.eval_on_split == "test":
            dataset_names = ["detect_test"]
        elif args.eval_on_split == "both":
            dataset_names = ["detect_test", "detect"]
        else:
            raise ValueError(f"Unknown eval_on_split {args.eval_on_split}")
    else:
        assert args.dataset_names is not None
        dataset_names = args.dataset_names

    for i, item in enumerate(available_trainings):
        logging.info(f"Evaluating {i+1}/{len(available_trainings)}: {item}")
        configs = tuple(product(dataset_names, args.weight_indices, args.random_seeds))
        for config in configs:
            eval_func(
                paths,
                item,
                config[0],
                config[1],
                create_plots=args.create_plots,
                random_seed=config[2],
                cuda=args.cuda,
                training_type=args.training_type,
                dataset_size=args.dataset_size,
                training_dir=training_dir,
                timing_only=args.timing_only,
            )


if __name__ == "__main__":
    flags = parse_arguments()
    logging.basicConfig(level=logging.INFO)
    evaluate_multiple(flags)
