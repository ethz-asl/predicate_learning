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

from typing import Dict

import torch
from torch import nn
import torch_geometric as pyg
import pytorch3d.transforms as py3d_trafo
from highlevel_planning_py.predicate_learning.training_gan import (
    TrainingSequenceGANctat,
    DatasetAdapterGraph,
)
from highlevel_planning_py.predicate_learning import training_utils as tu
from highlevel_planning_py.predicate_learning import data_utils as du


class DatasetAdapterPCEncoding(DatasetAdapterGraph):
    def __init__(self, device):
        super().__init__(device)
        self.device = device

    def evaluate_gt_classifier(
        self, batch, clf, dataset, get_reason=False, demo_ids=None
    ):
        batch_tensor, mask = pyg.utils.to_dense_batch(batch.x, batch.batch)
        manual_features = self.compute_manual_features(
            batch_tensor, batch, dataset.base_ds
        )
        if get_reason:
            return clf.check_reason(manual_features, mask, demo_ids)
        else:
            return clf.check(manual_features, mask, demo_ids)

    def compute_manual_features(self, batch_tensor, batch, dataset):
        manual_features = torch.zeros(
            (batch_tensor.size(0), batch_tensor.size(1), dataset.get_num_features())
        ).to(self.device)

        # Get argument features of demonstrations from dataset
        for i in range(batch.num_graphs):
            demo_id = batch.demo_id[i]
            features_args, features_others, _, _, _ = dataset.get_single_by_demo_id(
                demo_id, use_tensors=True
            )
            features_args.to(self.device)
            features_others.to(self.device)
            features_all = torch.cat((features_args, features_others), dim=0)

            num_objects = min(features_all.size(0), batch_tensor.size(1))
            manual_features[i, :num_objects, :] = features_all[:num_objects, :]

        # Set positions
        manual_features[:, :, :3] = batch_tensor[:, :, :3]

        # Set orientations
        base_orient = manual_features[:, :, (12, 9, 10, 11)]
        changed_orient = batch_tensor[:, :, (6, 3, 4, 5)]
        combined_orient = py3d_trafo.quaternion_multiply(changed_orient, base_orient)
        manual_features[:, :, 9:13] = combined_orient[:, :, (1, 2, 3, 0)]

        # Set AABB
        oabb = manual_features[:, :, 13:19]
        aabb = du.compute_aabb_torch(
            oabb, manual_features[:, :, :3], manual_features[:, :, 9:13], self.device
        )
        manual_features[:, :, 3:9] = aabb

        return manual_features

    def get_new_features_args(self, generated_scenes, dataset=None):
        manual_features_all = self.get_manual_features_vis(generated_scenes, dataset)
        manual_features_args = [feat[:2, :] for feat in manual_features_all]
        return manual_features_args

    def get_manual_features_vis(self, batch, dataset):
        batch_tensor, mask = pyg.utils.to_dense_batch(batch.x, batch.batch)
        mask = mask.cpu()
        manual_features = self.compute_manual_features(
            batch_tensor, batch, dataset.base_ds
        )
        manual_features = manual_features.detach().cpu().numpy()

        manual_features_list = list()
        for i in range(batch.num_graphs):
            features = manual_features[i, mask[i], :]
            manual_features_list.append(features)
        return manual_features_list


def create_data_adapters(loaders, device):
    adapters = dict()
    for label in loaders:
        adapters[label] = DatasetAdapterPCEncoding(device)
    return adapters


class TrainingSequenceGANctatPCEnc(TrainingSequenceGANctat):
    def __init__(
        self,
        run_config: tu.TrainingConfig,
        model_class_container: tu.ModelSplitter,
        model_disc_container: tu.ModelSplitter,
        model_gen: nn.Module,
        device,
        out_dir,
        loaders,
        paths: Dict,
        resume=False,
        debug=False,
        dataset=None,
        ground_truth_classifier=None,
    ):
        self.data_adapters = create_data_adapters(loaders, device)

        super().__init__(
            run_config,
            model_class_container,
            model_disc_container,
            model_gen,
            device,
            out_dir,
            loaders,
            paths,
            resume,
            debug,
            dataset,
            ground_truth_classifier,
        )

    def generate_vis(self):
        batch_size = self.data_adapters["gen"].get_batch_size(self.fixed_scene)
        generated_scenes, _ = self.generate(
            batch=self.fixed_scene, batch_size=batch_size, noise=self.fixed_latent
        )
        new_features_args = self.data_adapters["gen"].get_new_features_args(
            generated_scenes, self.dataset["train"]
        )
        return new_features_args
