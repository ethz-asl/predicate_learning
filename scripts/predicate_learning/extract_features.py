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
