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

from highlevel_planning_py.sim.scene_base import SceneBase
from highlevel_planning_py.sim.cupboard import get_cupboard_info


class SceneMoveSkill(SceneBase):
    def __init__(self, world, paths, restored_objects=None):
        SceneBase.__init__(self, world, paths, restored_objects)

        if restored_objects is None:
            self.objects = dict()
            self.objects["cupboard"] = get_cupboard_info(
                paths["asset_dir"], pos=[0.0, 2.0, 0.0], orient=[0.0, 0.0, 0.0, 1.0]
            )
            self.add_objects()
