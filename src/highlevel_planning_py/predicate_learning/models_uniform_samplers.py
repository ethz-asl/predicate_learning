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

import torch_geometric as pyg
import torch
from highlevel_planning_py.predicate_learning import data_utils
from highlevel_planning_py.predicate_learning import training_utils as tu
from highlevel_planning_py.predicate_learning.models_hybrid import (
    GANdiscriminatorHybrid,
)
from highlevel_planning_py.predicate_learning.models_sklearn import (
    DecisionTreeClassifier,
)


class UniformSampler:
    def __init__(self, config, paths):
        self.config = config

        if config.classifier_type == "none":
            self.classifier = None
        else:
            predicate_dir = os.path.join(paths["data_dir"], "predicates")
            training_dir = os.path.join(predicate_dir, "training", config.classifier_id)
            if config.classifier_type == "decisiontree":
                class_weights_file = "classifier_final.pkl"
                mdl_state = os.path.join(training_dir, class_weights_file)
                mdl_params = None
                model_class = DecisionTreeClassifier(mdl_params)
            else:
                available_class_weights = os.listdir(
                    os.path.join(training_dir, "models")
                )
                available_class_weights = [
                    item
                    for item in available_class_weights
                    if "class_model" in item and "final" not in item
                ]
                available_class_weights.sort()
                class_weights_file = available_class_weights[-1]
                mdl_params = tu.load_parameters(training_dir, "parameters_class")
                mdl_params.device = "cpu"
                state_file = os.path.join(training_dir, "models", class_weights_file)
                mdl_state = torch.load(state_file)
                model_class = GANdiscriminatorHybrid(mdl_params)
            model_class.load_state_dict(mdl_state)
            self.classifier = model_class

    def load_state_dict(self, state_dict):
        pass

    def to(self, device):
        pass

    def eval(self):
        pass

    def forward(self, batch_in, latent, desired_label=None):
        batch = batch_in.clone()
        dense_batch, _ = pyg.utils.to_dense_batch(batch.x, batch.batch)

        bounds = torch.zeros(
            (batch.num_graphs, 2, 7)
        )  # Contains lower and upper bounds
        if self.config.feature_type == "manual":
            assert self.config.feature_version == "v1"
            arg0_aabb = dense_batch[:, 0, 3:9]
            arg0_max_extent = torch.max(dense_batch[:, 0, 16:19], dim=1).values
            arg0_max_extent = arg0_max_extent.unsqueeze(1).expand(-1, 3)
            bounds[:, 0, :3] = (
                arg0_aabb[:, :3] - self.config.bb_expansion_factor * arg0_max_extent
            )
            bounds[:, 1, :3] = (
                arg0_aabb[:, 3:] + self.config.bb_expansion_factor * arg0_max_extent
            )
            bounds[:, 0, 3:] = -1.0
            bounds[:, 1, 3:] = 1.0
        elif self.config.feature_type == "pcenc":
            raise NotImplementedError
        else:
            raise ValueError("Unknown feature type")

        # Sample positions and orientations
        outputs_not_finished = list(range(batch.num_graphs))
        all_outputs = torch.zeros(batch.num_graphs, 1, 7)
        retries = 0
        while retries < self.config.max_iterations:
            sampler = torch.distributions.uniform.Uniform(
                bounds[outputs_not_finished, 0, :], bounds[outputs_not_finished, 1, :]
            )
            outputs = sampler.sample().unsqueeze(1)
            all_outputs[outputs_not_finished, :, :] = outputs

            norms = torch.linalg.norm(all_outputs[:, :, 3:], dim=2, keepdim=True) + 1e-8
            norms = norms.reshape(batch.num_graphs, -1, 1).expand(
                batch.num_graphs, -1, 4
            )
            normalized_quat = torch.div(all_outputs[:, :, 3:], norms)

            if self.config.feature_type == "manual":
                oabb = dense_batch[:, 1, 13:19].unsqueeze(1)
                aabb = data_utils.compute_aabb_torch(
                    oabb, all_outputs[:, :, :3], normalized_quat, "cpu"
                )
                completed_outputs = torch.cat(
                    (all_outputs[:, :, :3], aabb, normalized_quat, oabb), dim=2
                )
            else:
                completed_outputs = torch.cat(
                    (
                        all_outputs[:, :, :3],
                        normalized_quat,
                        dense_batch[:, 1, 7:].unsqueeze(1),
                    ),
                    dim=2,
                )

            batch.x[batch.ptr[:-1] + 1, :] = completed_outputs[:, 0, :]

            if self.classifier is None:
                break

            # Classify
            class_output = self.classifier.forward(batch)
            if self.config.classifier_type == "decisiontree":
                class_output = class_output.squeeze().bool()
            else:
                class_output = torch.sigmoid(class_output.squeeze()) > 0.5
            successful_indices = torch.where(class_output)[0]
            for i in successful_indices:
                if i in outputs_not_finished:
                    outputs_not_finished.remove(i)

            if len(outputs_not_finished) == 0:
                break

            retries += 1

        return batch
