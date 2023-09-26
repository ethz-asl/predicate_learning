import os
import igibson
from highlevel_planning_py.predicate_learning.predicate_learning_server import SimServer
from highlevel_planning_py.tools_pl import util
from highlevel_planning_py.tools_pl.logger_stdout import LoggerStdout
from highlevel_planning_py.tools_pl.path import get_path_dict

PATHS_SELECTOR = "local"


def main():
    flags = util.parse_arguments()
    flags.method = "direct"
    logger = LoggerStdout()
    paths = get_path_dict(PATHS_SELECTOR)
    sim_server = SimServer(
        flags, logger, paths, data_session_id="230606-124817_demonstrations_features"
    )
    pred_name = "inside_drawer"

    # Features
    # sim_server.pfm.extract_demo_features(pred_name, override_feature_version="v1")
    # sim_server.pfm.extract_demo_features(
    #     pred_name + "_test", override_feature_version="v1"
    # )
    # sim_server.pfm.extract_demo_features(pred_name, override_feature_version="v2")
    # sim_server.pfm.extract_demo_features(pred_name, override_feature_version="v3")
    # sim_server.pfm.extract_demo_features(
    #     pred_name + "_test", override_feature_version="v2"
    # )
    # sim_server.pfm.extract_demo_features(
    #     pred_name + "_test", override_feature_version="v3"
    # )

    # Point clouds
    # sim_server.pfm.extract_point_clouds(pred_name, num_points=2048)
    sim_server.pfm.extract_single_pointclouds(pred_name, num_points=2048)
    # sim_server.pfm.extract_point_clouds(pred_name + "_test", num_points=2048)
    # sim_server.pfm.extract_single_pointclouds(pred_name + "_test", num_points=2048)


if __name__ == "__main__":
    main()
