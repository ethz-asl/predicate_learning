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
