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

import torch
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from highlevel_planning_py.predicate_learning.data_utils import (
    shift_oabb,
    compute_aabb_torch,
)
from highlevel_planning_py.sim.world import WorldPybullet
import highlevel_planning_py.predicate_learning.groundtruth_tools as gt_tools


class ManualClassifier_On_AABB(gt_tools.ManualClassifier_Base):
    def __init__(self, above_tol: float, device: str, feature_version: str):
        super().__init__()
        self.above_tol = above_tol
        self.device = device
        self.feature_version = feature_version

    def check_reason_impl(self, features: torch.tensor, mask: torch.tensor, demo_ids):
        above_res = gt_tools.check_above(
            features, self.above_tol, self.feature_version, self.device
        )

        features_supporting = features[:, 0, :]
        features_supported = features[:, 1, :]
        pos_supported = features_supported[:, :3]
        if self.feature_version == "v1":
            lower_supporting = features_supporting[:, 3:6]
            upper_supporting = features_supporting[:, 6:9]
            aabbs = features[:, :, 3:9]
        elif self.feature_version in ["v2", "v3"]:
            # Compute aabb
            oabbs = (
                features[:, :, 19:25]
                if self.feature_version == "v2"
                else features[:, :, 16:22]
            )
            pos = features[:, :, :3]
            orient = features[:, :, 3:7]
            aabbs = compute_aabb_torch(oabbs, pos, orient, self.device)
            lower_supporting = aabbs[:, 0, :3]
            upper_supporting = aabbs[:, 0, 3:]
        else:
            raise ValueError

        gt_min = torch.ge(pos_supported[:, :2], lower_supporting[:, :2])
        gt_min = torch.logical_and(gt_min[:, 0], gt_min[:, 1])
        lt_max = torch.le(pos_supported[:, :2], upper_supporting[:, :2])
        lt_max = torch.logical_and(lt_max[:, 0], lt_max[:, 1])
        within = torch.logical_and(gt_min, lt_max)

        # Check if AABBs overlap
        aabb_overlap = gt_tools.check_aabbs_overlap(aabbs, self.device)

        on = torch.logical_and(above_res[0], within).float()

        reason = (*above_res[1], within.int(), aabb_overlap.int())

        return on, reason


class ManualClassifier_On_OABB(gt_tools.ManualClassifier_Base):
    def __init__(self, above_tol: float, device: str, feature_version: str):
        super().__init__()
        self.above_tol = above_tol
        self.device = device
        self.feature_version = feature_version

    def check_reason_impl(self, features: torch.tensor, mask: torch.tensor, demo_ids):
        pos_supported = features[:, 1, :2]
        pos_supporting = torch.unsqueeze(features[:, 0, :3], dim=1)
        if self.feature_version == "v1":
            oabb_supporting = torch.unsqueeze(features[:, 0, 13:19], dim=1)
            orient_supporting = torch.unsqueeze(features[:, 0, 9:13], dim=1)
            aabbs = features[:, :, 3:9]
        elif self.feature_version == "v2":
            oabb_supporting = torch.unsqueeze(features[:, 0, 19:25], dim=1)
            orient_supporting = torch.unsqueeze(features[:, 0, 3:7], dim=1)
            aabbs = None
        elif self.feature_version == "v3":
            oabb_supporting = torch.unsqueeze(features[:, 0, 16:22], dim=1)
            orient_supporting = torch.unsqueeze(features[:, 0, 3:7], dim=1)
            aabbs = None
        else:
            raise ValueError
        shifted_oabb = shift_oabb(
            oabb_supporting, pos_supporting, orient_supporting, self.device
        )
        shifted_oabb = shifted_oabb[:, :, 4:, :2]

        within = torch.ones(features.size(0)).to(self.device)
        for i in range(features.size(0)):
            point = Point(pos_supported[i, 0], pos_supported[i, 1])
            polygon = Polygon(
                [
                    (shifted_oabb[i, 0, 0, 0], shifted_oabb[i, 0, 0, 1]),
                    (shifted_oabb[i, 0, 2, 0], shifted_oabb[i, 0, 2, 1]),
                    (shifted_oabb[i, 0, 3, 0], shifted_oabb[i, 0, 3, 1]),
                    (shifted_oabb[i, 0, 1, 0], shifted_oabb[i, 0, 1, 1]),
                ]
            )
            within[i] = polygon.contains(point)

        above_res = gt_tools.check_above(
            features, self.above_tol, self.feature_version, self.device
        )

        if aabbs is None:
            aabb_overlap = torch.zeros(features.size(0)).to(self.device)
        else:
            aabb_overlap = gt_tools.check_aabbs_overlap(aabbs, self.device)

        result = torch.logical_and(above_res[0], within).float()
        reason = (*above_res[1], within.int(), aabb_overlap.int())

        return result, reason


class ManualClassifier_OnCfree_OABB(ManualClassifier_On_OABB):
    def __init__(
        self,
        above_tol: float,
        device: str,
        feature_version: str,
        always_check_collision=False,
    ):
        super().__init__(above_tol, device, feature_version)
        self.always_check_collision = always_check_collision
        self.w = WorldPybullet(style="direct", include_floor=False)

    def check_reason_impl(
        self, features: torch.tensor, mask: torch.tensor, demo_ids
    ) -> torch.tensor:
        on, reason_on = super().check_reason_impl(features, mask, demo_ids)

        prerequisite = None if self.always_check_collision else on

        cfree = gt_tools.check_cfree(
            features, mask, prerequisite, self.device, self.w, self.feature_version
        )

        result = torch.logical_and(on.bool(), cfree.bool()).float()

        # Bring reason into correct format
        reason_overall = (*reason_on, cfree.int(), result.int())
        reason_overall = gt_tools.stringify_reason(reason_overall)

        return result, reason_overall
