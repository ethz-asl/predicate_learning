import os

import torch
import highlevel_planning_py.predicate_learning.data_utils as du
import highlevel_planning_py.predicate_learning.groundtruth_tools as gt_tools
from highlevel_planning_py.sim.world import WorldPybullet
from highlevel_planning_py.predicate_learning.features import PredicateFeatureManager
from highlevel_planning_py.sim.fake_perception import FakePerceptionPipeline
from highlevel_planning_py.sim.scene_base import SceneBase
from highlevel_planning_py.tools_pl.logger_stdout import LoggerStdout


class ManualClassifier_Inside(gt_tools.ManualClassifier_Base):
    def __init__(
        self,
        device: str,
        feature_version: str,
        pen_tolerance: float,
        world: WorldPybullet = None,
        debug: bool = False,
    ):
        super().__init__()
        self.device = device
        self.feature_version = feature_version
        self.pen_tolerance = pen_tolerance
        if feature_version != "v1":
            raise NotImplementedError
        if world is None:
            if debug:
                self.world = WorldPybullet(style="gui")
            else:
                self.world = WorldPybullet(style="direct")
        else:
            self.world = world
        self.collision_checker = None

    def check_reason_impl(self, features: torch.tensor, mask: torch.tensor, demo_ids):
        features_container = features[:, 0, :]
        features_contained = features[:, 1, :]

        oabb_contained = features_contained[:, 13:19].unsqueeze(1)
        pos_contained = features_contained[:, 0:3].unsqueeze(1)
        orient_contained = features_contained[:, 9:13].unsqueeze(1)
        pos_container = features_container[:, 0:3].unsqueeze(1)
        orient_container = features_container[:, 9:13].unsqueeze(1)
        oabb_container = features_container[:, 13:19].unsqueeze(1)

        oabb_contained_corners = du.get_vertices_from_oabb(
            oabb_contained, pos_contained.dtype
        ).to(self.device)
        oabb_contained_corners = du.apply_rotation_to_vertices(
            oabb_contained_corners, orient_contained
        )
        oabb_contained_corners = (
            oabb_contained_corners
            + pos_contained.unsqueeze(2).expand_as(oabb_contained_corners)
            - pos_container.unsqueeze(2).expand_as(oabb_contained_corners)
        )
        oabb_contained_corners = du.apply_rotation_to_vertices(
            oabb_contained_corners, orient_container, inv=True
        )
        oabb_contained_corners = oabb_contained_corners.squeeze(1)

        not_too_high = torch.all(
            oabb_contained_corners[:, :, 2] - oabb_container[:, :, 5] < 0, dim=1
        )
        not_too_low = torch.all(
            oabb_contained_corners[:, :, 2] - oabb_container[:, :, 2] > 0, dim=1
        )
        not_outside_x = torch.all(
            torch.abs(oabb_contained_corners[:, :, 0]) - oabb_container[:, :, 3] < 0,
            dim=1,
        )
        not_outside_y = torch.all(
            torch.abs(oabb_contained_corners[:, :, 1]) - oabb_container[:, :, 4] < 0,
            dim=1,
        )

        res = not_too_high & not_too_low & not_outside_x & not_outside_y

        # Check for collisions
        cfree = self.collision_checker(features, mask, res, demo_ids).bool()
        res = res & cfree

        res = res.float()
        reason = (
            not_too_high.int(),
            not_too_low.int(),
            not_outside_x.int(),
            not_outside_y.int(),
            cfree.int(),
            res.int(),
        )
        reason = gt_tools.stringify_reason(reason)

        return res, reason

    def close(self):
        self.world.close()


class ManualClassifier_Inside_OABB(ManualClassifier_Inside):
    def __init__(
        self,
        device: str,
        feature_version: str,
        pen_tolerance: float,
        world: WorldPybullet = None,
    ):
        super().__init__(device, feature_version, pen_tolerance, world)
        self.collision_checker = lambda features, mask, prerequisite, demo_ids: gt_tools.check_cfree(
            features,
            mask,
            prerequisite,
            device,
            self.world,
            feature_version,
            pen_tolerance,
        )


class ManualClassifier_Inside_OABB_TrueGeom(ManualClassifier_Inside):
    def __init__(
        self,
        device: str,
        feature_version: str,
        pen_tolerance: float,
        paths: dict,
        data_session_id,
        world: WorldPybullet = None,
        debug: bool = False,
    ):
        super().__init__(device, feature_version, pen_tolerance, world, debug)

        logger_stdout = LoggerStdout(level="warning")
        pred_data_dir = os.path.join(paths["data_dir"], "predicates", "data")

        self.scene = SceneBase(self.world, paths)
        self.fpp = FakePerceptionPipeline(logger_stdout, self.world.client_id, paths)
        self.pfm = PredicateFeatureManager(
            pred_data_dir,
            data_session_id,
            self.world,
            self.scene,
            self.fpp,
            None,
            logger_stdout,
            paths,
        )

        pred_name = "inside_drawer"
        feature_dir = os.path.join(
            pred_data_dir, pred_name, "features", data_session_id
        )
        filename = os.path.join(feature_dir, f"{pred_name}_{feature_version}.pkl")
        _, relations, _ = self.pfm.load_existing_features(filename)
        pred_name_ = "inside_drawer_test"
        feature_dir = os.path.join(
            pred_data_dir, pred_name_, "features", data_session_id
        )
        filename = os.path.join(feature_dir, f"{pred_name_}_{feature_version}.pkl")
        _, relations_test, _ = self.pfm.load_existing_features(filename)
        relations.update(relations_test)

        self.collision_checker = lambda features, mask, prerequisite, demo_ids: gt_tools.check_cfree_true_geom(
            features,
            mask,
            prerequisite,
            device,
            self.world,
            feature_version,
            demo_ids,
            self.pfm,
            pred_name,
            relations,
            relations_test,
            pen_tolerance,
        )


if __name__ == "__main__":
    gt = ManualClassifier_Inside_OABB("cpu", "v1", 0.03)
    fake_features = torch.randn(16, 3, 20)
    gt.check_reason(fake_features, None)
