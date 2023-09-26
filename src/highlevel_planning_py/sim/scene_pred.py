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
import os
from highlevel_planning_py.tools_pl.util import ObjectInfo
from highlevel_planning_py.sim.scene_base import SceneBase
import igibson


class ScenePred(SceneBase):
    def __init__(self, world, base_dir, restored_objects=None):
        SceneBase.__init__(self, world, base_dir, restored_objects)

        if restored_objects is None:
            cabinet_lower_path = os.path.join(
                igibson.assets_path, "models/cabinet2/cabinet_0007.urdf"
            )
            self.objects["cabinet_lower"] = ObjectInfo(
                urdf_name_="cabinet_0007.urdf",
                urdf_path_=cabinet_lower_path,
                init_pos_=np.array([-0.5, 0, 0.5]),
                init_orient_=np.array([0, 0, 0, 1]),
                merge_fixed_links_=True,
            )

            cabinet_upper_path = os.path.join(
                igibson.assets_path, "models/cabinet/cabinet_0004.urdf"
            )
            self.objects["cabinet_upper"] = ObjectInfo(
                urdf_name_="cabinet_0004.urdf",
                urdf_path_=cabinet_upper_path,
                init_pos_=np.array([1.0, 0, 0.5]),
                init_orient_=np.array([0, 0, 0, 1]),
                merge_fixed_links_=True,
            )

            # Find available objects
            ycb_dir = os.path.join(igibson.assets_path, "models", "ycb")
            available_objects_ = os.listdir(ycb_dir)
            available_objects = list()
            for obj in available_objects_:
                if os.path.isdir(os.path.join(ycb_dir, obj)):
                    available_objects.append(obj)
            del available_objects_
            available_objects.sort()

            for i in range(2):
                for j in range(3):
                    new_object = available_objects[i * 3 + j]
                    init_pos = np.array([0.6 + 0.3 * j, 1.5 + 0.3 * i, 0.3])
                    self.objects[f"ycb_{i*3+j}"] = ObjectInfo(
                        urdf_name_=new_object,
                        urdf_path_=os.path.join(ycb_dir, new_object),
                        init_pos_=init_pos,
                        init_orient_=np.array([0, 0, 0, 1]),
                        init_scale_=1.0,
                    )

            self.add_objects()
