import time
import torch
import pybullet as pb
from scipy.spatial.transform import Rotation

from highlevel_planning_py.predicate_learning.features import PredicateFeatureManager
from highlevel_planning_py.sim.world import WorldPybullet
from shapely.geometry.polygon import Polygon
from highlevel_planning_py.predicate_learning.data_utils import compute_aabb_torch

# import matplotlib.pyplot as plt


def check_above(features, above_tol, feature_version, device):
    if feature_version == "v1":
        features_supporting = features[:, 0, :]
        features_supported = features[:, 1, :]
        upper_supporting = features_supporting[:, 6:9]
        lower_supported = features_supported[:, 3:6]
    elif feature_version in ["v2", "v3"]:
        # Compute aabb
        oabbs = (
            features[:, :, 19:25] if feature_version == "v2" else features[:, :, 16:22]
        )
        pos = features[:, :, :3]
        orient = features[:, :, 3:7]
        aabbs = compute_aabb_torch(oabbs, pos, orient, device)
        upper_supporting = aabbs[:, 0, 3:]
        lower_supported = aabbs[:, 1, :3]
    else:
        raise ValueError

    not_too_low = torch.lt(upper_supporting[:, 2] - above_tol, lower_supported[:, 2])
    not_too_high = torch.lt(lower_supported[:, 2], upper_supporting[:, 2] + above_tol)
    above = torch.logical_and(not_too_low, not_too_high)

    return above, (not_too_low.int(), not_too_high.int())


def check_aabbs_overlap(aabbs, device):
    aabb_overlap = torch.zeros(aabbs.shape[0]).bool().to(device)
    for i in range(aabbs.shape[0]):
        supporting_polygon = Polygon(
            [
                (aabbs[i, 0, 0], aabbs[i, 0, 1]),
                (aabbs[i, 0, 0], aabbs[i, 0, 4]),
                (aabbs[i, 0, 3], aabbs[i, 0, 4]),
                (aabbs[i, 0, 3], aabbs[i, 0, 1]),
            ]
        )
        supported_polygon = Polygon(
            [
                (aabbs[i, 1, 0], aabbs[i, 1, 1]),
                (aabbs[i, 1, 0], aabbs[i, 1, 4]),
                (aabbs[i, 1, 3], aabbs[i, 1, 4]),
                (aabbs[i, 1, 3], aabbs[i, 1, 1]),
            ]
        )
        aabb_overlap[i] = supporting_polygon.intersects(supported_polygon)

        # x, y = supporting_polygon.exterior.xy
        # plt.plot(x, y)
        # x, y = supported_polygon.exterior.xy
        # plt.plot(x, y)
        # plt.show()
    return aabb_overlap


class ManualClassifier_Base:
    def __init__(self):
        self.total_time = 0.0

    def check(
        self, features: torch.tensor, mask: torch.tensor, demo_ids=None
    ) -> torch.tensor:
        res, _ = self.check_reason(features, mask, demo_ids)
        return res

    def check_reason(self, features: torch.tensor, mask: torch.tensor, demo_ids=None):
        start = time.time()
        res, reason = self.check_reason_impl(features, mask, demo_ids)
        end = time.time()
        self.total_time += end - start
        return res, reason

    def check_reason_impl(self, features: torch.tensor, mask: torch.tensor, demo_ids):
        raise NotImplementedError

    def close(self):
        pass


def stringify_reason(reason):
    reason_overall = torch.stack(reason, dim=1).tolist()
    reason_overall = [[str(it) for it in li] for li in reason_overall]
    reason_overall = ["".join(li) for li in reason_overall]
    return reason_overall


def check_cfree(
    features,
    mask,
    prerequisite,
    device,
    world: WorldPybullet,
    feature_version,
    pen_tolerance=0.0,
    demo_ids=None,
):
    positions = features[:, :, :3]
    if feature_version == "v1":
        oabbs = features[:, :, 13:19]
        orients = features[:, :, 9:13]
    elif feature_version == "v2":
        oabbs = features[:, :, 19:25]
        orients = features[:, :, 3:7]
    elif feature_version == "v3":
        oabbs = features[:, :, 16:22]
        orients = features[:, :, 3:7]
    else:
        raise ValueError

    world.reset()

    num_scenes = features.size(0)
    num_objects = features.size(1)
    cfree = torch.ones(num_scenes).to(device)
    uids = list()
    for i_scene in range(num_scenes):
        if prerequisite is not None and not prerequisite[i_scene]:
            # We don't even need to check this
            continue

        found_collision = False

        # Check collisions between supported object and all other objects
        for i_obj1 in range(1, 2):
            if mask[i_scene, i_obj1].item() < 0.5:
                continue
            for i_obj2 in range(i_obj1 + 1, num_objects):
                if mask[i_scene, i_obj2].item() < 0.5:
                    continue

                for uid in uids:
                    pb.removeBody(uid, physicsClientId=world.client_id)
                uids.clear()

                # Create objects in pb
                for i_obj in [i_obj1, i_obj2]:
                    pos = positions[i_scene, i_obj, :].tolist()
                    pos[2] += 1.0
                    orient = orients[i_scene, i_obj, :].tolist()
                    half_extents = oabbs[i_scene, i_obj, 3:].tolist()
                    cid = pb.createCollisionShape(
                        shapeType=pb.GEOM_BOX,
                        halfExtents=half_extents,
                        physicsClientId=world.client_id,
                    )
                    # vid = pb.createVisualShape(
                    #     shapeType=pb.GEOM_BOX,
                    #     halfExtents=half_extents,
                    #     physicsClientId=self.w.client_id,
                    # )
                    uid = pb.createMultiBody(
                        baseCollisionShapeIndex=cid,
                        # baseVisualShapeIndex=vid,
                        basePosition=pos,
                        baseOrientation=orient,
                        physicsClientId=world.client_id,
                    )
                    uids.append(uid)

                # Check collision
                temp = pb.getClosestPoints(
                    uids[0], uids[1], distance=0.05, physicsClientId=world.client_id
                )
                for elem in temp:
                    contact_distance = elem[8]
                    if contact_distance < 0.0 - pen_tolerance:
                        cfree[i_scene] = 0.0
                        found_collision = True
                        break

                if found_collision:
                    break
            if found_collision:
                break
    for uid in uids:
        pb.removeBody(uid, physicsClientId=world.client_id)
    uids.clear()
    return cfree


def place_object(obj_features, obj_name, pfm, client_id, feature_version):
    obj_ids = pfm.outer_perception.object_info["object_ids_by_name"][obj_name]
    obj_info = pfm.outer_perception.object_info["objects_by_id"][obj_ids]

    # Base position and orientation
    if feature_version == "v1":
        orient = Rotation.from_quat(obj_features[9:13].tolist())
    else:
        raise NotImplementedError
    com_pos = obj_info.centroid
    base_pos = obj_features[:3].tolist() - orient.apply(com_pos)

    # Move object
    uid = obj_ids[0]
    pb.resetBasePositionAndOrientation(
        uid, base_pos, orient.as_quat(), physicsClientId=client_id
    )


def check_cfree_true_geom(
    features,
    mask,
    prerequisite,
    device,
    world: WorldPybullet,
    feature_version,
    demo_ids,
    predicate_feature_manager: PredicateFeatureManager,
    pred_name,
    relations,
    relations_test,
    pen_tolerance=0.0,
):
    pfm = predicate_feature_manager
    num_scenes = features.size(0)
    cfree = torch.ones(num_scenes).to(device)
    for i_scene in range(num_scenes):
        if prerequisite is not None and not prerequisite[i_scene]:
            # We don't even need to check this
            continue

        # Load demo in simulator
        if demo_ids[i_scene] not in relations_test:
            pfm.restore_demonstration_outside(pred_name, demo_ids[i_scene])
        else:
            pfm.restore_demonstration_outside(pred_name + "_test", demo_ids[i_scene])

        # Move objects into correct positions
        num_arg_objects = len(relations[demo_ids[i_scene]]["argument_names"])
        for i_obj, obj_name in enumerate(
            relations[demo_ids[i_scene]]["argument_names"]
        ):
            place_object(
                features[i_scene, i_obj, :],
                obj_name,
                pfm,
                world.client_id,
                feature_version,
            )
        for i_obj, obj_name in enumerate(relations[demo_ids[i_scene]]["other_names"]):
            place_object(
                features[i_scene, i_obj + num_arg_objects, :],
                obj_name,
                pfm,
                world.client_id,
                feature_version,
            )

        found_collision = False

        # Check collisions between supported object and all other objects
        uid_arg_obj = pfm.outer_perception.object_info["object_ids_by_name"][
            relations[demo_ids[i_scene]]["argument_names"][1]
        ][0]
        for other_obj_name in relations[demo_ids[i_scene]]["other_names"]:
            uid_other_obj = pfm.outer_perception.object_info["object_ids_by_name"][
                other_obj_name
            ][0]
            temp = pb.getClosestPoints(
                uid_arg_obj,
                uid_other_obj,
                distance=0.01,
                physicsClientId=world.client_id,
            )
            for elem in temp:
                contact_distance = elem[8]
                if contact_distance < 0.0 - pen_tolerance:
                    cfree[i_scene] = 0.0
                    found_collision = True
                    break

            if found_collision:
                break
    return cfree
