import os
import sys
import argparse
import logging
from copy import deepcopy

from highlevel_planning_py.predicate_learning import training_utils as tu
from highlevel_planning_py.tools_pl.path import get_path_dict


def parse_arguments():
    parser = argparse.ArgumentParser()
    tu.parse_general_args(parser)
    parser.add_argument("--feature_version", type=str, default="v1")
    parser.add_argument("--data_normalization_class", type=str, default="none")
    parser.add_argument("--feature_type", type=str, choices=["manual", "pcenc"])
    parser.add_argument("--encoder_id", type=str, default="")
    parser.add_argument("--bb_expansion_factor", type=float, default=0.5)
    parser.add_argument(
        "--classifier_type", type=str, choices=["none", "gan", "pcgan", "decisiontree"]
    )
    parser.add_argument("--classifier_id", type=str, default="")
    parser.add_argument("--description", type=str, default=None)
    parser.add_argument("--max_iterations", type=int, default=50)
    args = parser.parse_args()
    return args


def main(args):
    if args.feature_type == "manual":
        assert args.classifier_type in ["gan", "decisiontree", "none"]
    elif args.feature_type == "pcenc":
        assert args.classifier_type in ["pcgan", "decisiontree", "none"]
    else:
        raise ValueError

    paths = get_path_dict(args.paths)
    predicate_dir = os.path.join(paths["data_dir"], "predicates")
    out_dir = tu.setup_out_dir(
        predicate_dir,
        resume_training=False,
        resume_timestring=args.id_string,
        screen_output=args.screen_output,
        training_type="sampler"
        if args.description is None
        else f"sampler_{args.description}",
    )
    device = "cpu"
    logging.info(f"Command: {' '.join(sys.argv)}")
    tu.save_parameters(vars(args), "parameters_training", out_dir, parameters_obj=args)
    with open(os.path.join(out_dir, "dummy.txt"), "w") as f:
        f.write("dummy")

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


if __name__ == "__main__":
    arguments = parse_arguments()
    main(arguments)
