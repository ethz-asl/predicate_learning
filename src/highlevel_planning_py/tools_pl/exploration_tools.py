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

import pybullet as p
import numpy as np


def get_items_closeby(
    goal_objects, scene_objects, pb_client_id, robot_uid=None, distance_limit=0.5
):
    closeby_objects = set()
    for obj in scene_objects:
        if obj in goal_objects:
            continue

        obj_uid = scene_objects[obj].model.uid

        if robot_uid is not None:
            ret = p.getClosestPoints(
                robot_uid,
                obj_uid,
                distance=1.2 * distance_limit,
                physicsClientId=pb_client_id,
            )
            if len(ret) > 0:
                distances = np.array([r[8] for r in ret])
                distance = np.min(distances)
                if distance <= distance_limit:
                    closeby_objects.add(obj)
                    continue

        for goal_obj in goal_objects:
            goal_obj_uid = scene_objects[goal_obj].model.uid
            ret = p.getClosestPoints(
                obj_uid,
                goal_obj_uid,
                distance=1.2 * distance_limit,
                physicsClientId=pb_client_id,
            )
            if len(ret) == 0:
                continue
            distances = np.array([r[8] for r in ret])
            distance = np.min(distances)
            if distance <= distance_limit:
                closeby_objects.add(obj)
    return list(closeby_objects)
