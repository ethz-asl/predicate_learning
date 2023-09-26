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

import numpy as np
from highlevel_planning_py.tools_pl.util import rotate_orient, ObjectInfo
from highlevel_planning_py.sim.scene_base import SceneBase
from highlevel_planning_py.sim.cupboard import get_cupboard_info

from scipy.spatial.transform import Rotation as R


class ScenePlanning2(SceneBase):
    def __init__(self, world, paths, restored_objects=None):
        SceneBase.__init__(self, world, paths, restored_objects)

        if restored_objects is None:
            self.objects["table"] = ObjectInfo(
                urdf_path_="table/table.urdf",
                urdf_relative_to_="asset_dir",
                init_pos_=np.array([3.0, 0.0, 0.0]),
                init_orient_=np.array([0.0, 0.0, 0.0, 1.0]),
                force_fixed_base_=True,
            )
            self.objects["cube1"] = ObjectInfo(
                urdf_path_="cube_small.urdf",
                init_pos_=np.array([2.5, 0.0, 0.7]),
                # init_orient_=np.array([0.0, 0.0, 0.0, 1.0]),
                init_orient_=rotate_orient(np.array([0.0, 0.0, 0.0, 1.0]), "z", -20.0),
                grasp_links_=[-1],
                grasp_pos_={-1: [np.array([0.0, 0.0, 0.0])]},
                grasp_orient_={-1: [np.array([0.0, 0.0, 0.0, 1.0])]},
            )
            self.objects["lego"] = ObjectInfo(
                urdf_path_="lego/lego.urdf",
                init_pos_=np.array([2.45, -0.3, 0.68]),
                init_orient_=rotate_orient(np.array([0.0, 0.0, 0.0, 1.0]), "z", 0.0),
                init_scale_=2.0,
                grasp_links_=[-1],
                grasp_pos_={-1: [np.array([0.0, 0.0, 0.0])]},
                grasp_orient_={-1: [np.array([0.0, 0.0, 0.0, 1.0])]},
            )
            self.objects["duck"] = ObjectInfo(
                urdf_path_="duck_vhacd.urdf",
                init_pos_=np.array([2.8, -0.25, 0.67]),
                # init_pos_=np.array([3.55, -0.23, 0.7]),
                init_orient_=rotate_orient(np.array([0.0, 0.0, 0.0, 1.0]), "x", 90.0),
                init_scale_=0.8,
                grasp_links_=[-1],
                grasp_pos_={-1: [np.array([-0.015, 0.03, 0.0])]},
                grasp_orient_={
                    -1: [rotate_orient(np.array([0.0, 0.0, 0.0, 1.0]), "x", -90.0)]
                },
            )
            self.objects["container1"] = ObjectInfo(
                urdf_path_="parsed_xacros/container_no_lid.urdf",
                urdf_relative_to_="asset_dir",
                init_pos_=np.array([3.5, -0.25, 0.625]),
                init_orient_=np.array([0.0, 0.0, 0.0, 1.0]),
                force_fixed_base_=True,
            )
            self.objects["lid1"] = ObjectInfo(
                urdf_path_="parsed_xacros/lid.urdf",
                urdf_relative_to_="asset_dir",
                init_pos_=np.array([3.5, -0.25, 0.775]),
                init_orient_=np.array([0.0, 0.0, 0.0, 1.0]),
                grasp_pos_={5: [np.array([0.0, 0.0, 0.0])]},
                grasp_orient_={
                    5: [rotate_orient(np.array([0.0, 0.0, 0.0, 1.0]), "y", 90)]
                },
                grasp_links_=[5],
                friction_setting_=[{"link_name": "handle", "lateral_friction": 1.0}],
            )
            # self.objects["cube2"] = ObjectInfo(
            #     urdf_path_="cube_small.urdf",
            #     init_pos_=np.array([3.55, -0.23, 0.7]),
            #     # init_orient_=np.array([0.0, 0.0, 0.0, 1.0]),
            #     init_orient_=rotate_orient(np.array([0.0, 0.0, 0.0, 1.0]), "z", -20.0),
            #     grasp_links_=[-1],
            #     grasp_pos_={-1: [np.array([0.0, 0.0, 0.0])]},
            #     grasp_orient_={-1: [np.array([0.0, 0.0, 0.0, 1.0])]},
            # )
            self.objects["container2"] = ObjectInfo(
                urdf_path_="parsed_xacros/container_no_lid.urdf",
                urdf_relative_to_="asset_dir",
                init_pos_=np.array([3.5, 0.25, 0.625]),
                init_orient_=np.array([0.0, 0.0, 0.0, 1.0]),
                force_fixed_base_=True,
            )
            self.objects["lid2"] = ObjectInfo(
                urdf_path_="parsed_xacros/lid.urdf",
                urdf_relative_to_="asset_dir",
                init_pos_=np.array([3.5, 0.25, 0.775]),
                init_orient_=np.array([0.0, 0.0, 0.0, 1.0]),
                grasp_pos_={5: [np.array([0.0, 0.0, 0.0])]},
                grasp_orient_={
                    5: [rotate_orient(np.array([0.0, 0.0, 0.0, 1.0]), "y", 90)]
                },
                grasp_links_=[5],
                friction_setting_=[{"link_name": "handle", "lateral_friction": 1.0}],
            )
            self.objects["can"] = ObjectInfo(
                urdf_path_="coke_can/model.sdf",
                urdf_relative_to_="asset_dir",
                init_pos_=np.array([3.0, 0.25, 0.7]),
                init_orient_=np.array([0.0, 0.0, 0.0, 1.0]),
                init_scale_=0.8,
                grasp_pos_={
                    -1: [np.array([0.0, 0.0, 0.03]), np.array([0.0, 0.0, 0.025])]
                },
                grasp_orient_={
                    -1: [
                        rotate_orient(np.array([0.0, 0.0, 0.0, 1.0]), "z", -90),
                        R.from_euler("y", -55, degrees=True).as_quat(),
                    ]
                },
                grasp_links_=[-1],
            )
            self.objects["tall_box"] = ObjectInfo(
                urdf_path_="tall_box.urdf",
                urdf_relative_to_="asset_dir",
                init_pos_=np.array([2.8, 0.25, 0.7]),
                init_orient_=R.from_euler("z", -20, degrees=True).as_quat(),
                init_scale_=1.0,
                grasp_pos_={
                    -1: [np.array([0.0, 0.0, 0.055]), np.array([0.0, 0.0, 0.05])]
                },
                grasp_orient_={
                    -1: [
                        rotate_orient(np.array([0.0, 0.0, 0.0, 1.0]), "z", -90),
                        R.from_euler("y", -80, degrees=True).as_quat(),
                    ]
                },
                grasp_links_=[-1],
                friction_setting_=[{"link_name": "base_link", "lateral_friction": 2.0}],
            )
            self.objects["shelf"] = ObjectInfo(
                urdf_path_="parsed_xacros/shelf.urdf",
                urdf_relative_to_="asset_dir",
                init_pos_=np.array([0.0, -1.5, 0.0]),
                init_orient_=R.from_euler("z", 90, degrees=True).as_quat(),
                nav_min_dist_=0.5,
                nav_angle_=np.pi / 4.0,
                force_fixed_base_=True,
            )
            self.objects["cupboard"] = get_cupboard_info(
                paths["asset_dir"],
                pos=[0.0, 1.5, 0.0],
                orient=[0.0, 0.0, 0.0, 1.0],
                force_fixed_base=True,
            )

            self.add_objects()
