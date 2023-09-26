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

import pickle

import torch
from sklearn import tree
import numpy as np
import torch_geometric as pyg

from highlevel_planning_py.predicate_learning import data_utils


BOUND_CAP = 2.0


class DecisionTreeModel:
    def __init__(self, config):
        self.config = config
        self.model = None

    def create_new_model(self):
        self.model = tree.DecisionTreeClassifier(
            criterion=self.config["criterion"],
            splitter=self.config["splitter"],
            max_depth=self.config["max_depth"],
        )

    def load_state_dict(self, state_dict):
        with open(state_dict, "rb") as f:
            self.model = pickle.load(f)

    def to(self, device):
        if device != "cpu":
            raise NotImplementedError("Only cpu is supported")

    def forward(self, **args):
        raise NotImplementedError


class DecisionTreeClassifier(DecisionTreeModel):
    def __init__(self, config):
        super().__init__(config)

    def forward(self, batch):
        x = batch.x.numpy().reshape(1, -1)
        res = self.model.predict(x)
        res = torch.Tensor(res).unsqueeze(0)
        return res


class DecisionTreeGenerator(DecisionTreeModel):
    def __init__(self, config):
        super().__init__(config)
        self.parents = None
        self.positive_children = None

    @staticmethod
    def sample_noise(batch_size, dimension):
        return None

    @staticmethod
    def eval():
        pass

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)

        # Reverse the tree
        tree = self.model.tree_
        self.parents = [None] * tree.node_count
        self.positive_children = list()
        for i in range(tree.node_count):
            if tree.children_left[i] != -1:
                self.parents[tree.children_left[i]] = i
            if tree.children_right[i] != -1:
                self.parents[tree.children_right[i]] = i
            if tree.children_left[i] == -1 and tree.children_right[i] == -1:
                if tree.value[i][0][1] > tree.value[i][0][0]:
                    self.positive_children.append(i)

    def forward(self, batch_in, latent, desired_label=None):
        batch = batch_in.clone()

        tree = self.model.tree_

        dense_batch, _ = pyg.utils.to_dense_batch(batch.x, batch.batch)

        num_features = batch.num_features
        out_indices_to_sample = 1
        start_idx = out_indices_to_sample * num_features
        if self.config.feature_type == "manual":
            if self.config.feature_version != "v1":
                raise NotImplementedError
            features_to_sample = list(range(start_idx, start_idx + 3)) + list(
                range(start_idx + 9, start_idx + 13)
            )
        elif self.config.feature_type == "pcenc":
            features_to_sample = list(range(start_idx, start_idx + 7))
        else:
            raise ValueError("Unknown feature type")
        features_being_resampled = list(range(start_idx, start_idx + num_features))

        outputs = torch.zeros(batch.num_graphs, 1, 7)
        for i in range(batch.num_graphs):
            single_graph = batch.get_example(i)
            input_features = single_graph.x[:2, :].numpy().reshape(1, -1)
            max_extent_arg1 = (
                np.max(input_features[0, start_idx + 16 : start_idx + 19])
                if self.config.feature_type == "manual"
                else None
            )

            # Choose a child to start from
            children_shuffled = np.random.permutation(self.positive_children)
            found_solution = False
            output_features_lb = None
            output_features_ub = None
            for child in children_shuffled:
                node = child
                parent = self.parents[node]
                output_features_lb = [-np.inf] * input_features.shape[1]
                output_features_ub = [np.inf] * input_features.shape[1]
                while parent is not None:
                    feature = tree.feature[parent]
                    threshold = tree.threshold[parent]

                    # Make sure this path complies with features that will stay the same
                    if feature not in features_being_resampled:
                        if node == tree.children_left[parent]:
                            # Left child
                            if not input_features[0, feature] <= threshold:
                                break
                        else:
                            # Right child
                            if not input_features[0, feature] > threshold:
                                break

                    if node == tree.children_left[parent]:
                        # Left child
                        output_features_ub[feature] = np.min(
                            (threshold, output_features_ub[feature])
                        )
                    else:
                        # Right child
                        output_features_lb[feature] = np.max(
                            (threshold, output_features_lb[feature])
                        )

                    node = parent
                    parent = self.parents[node]
                if node == 0:
                    found_solution = True
                    break
            output = None
            if found_solution:
                bounds = np.array([output_features_lb, output_features_ub])
                extended_bounds = np.zeros((2, len(features_to_sample)))

                # Positions based on position bounds
                extended_bounds[0, :3] = bounds[0, features_to_sample[:3]]
                extended_bounds[1, :3] = bounds[1, features_to_sample[:3]]

                # Positions based on bounding box bounds
                if self.config.feature_type == "manual":
                    extended_bounds[0, :3] = np.max(
                        (
                            extended_bounds[0, :3],
                            bounds[0, start_idx + 3 : start_idx + 6] + max_extent_arg1,
                        ),
                        axis=0,
                    )
                    extended_bounds[1, :3] = np.min(
                        (
                            extended_bounds[1, :3],
                            bounds[1, start_idx + 6 : start_idx + 9] - max_extent_arg1,
                        ),
                        axis=0,
                    )

                # Orientations based on orientation bounds
                extended_bounds[0, 3:] = bounds[0, features_to_sample[3:]]
                extended_bounds[1, 3:] = bounds[1, features_to_sample[3:]]

                # Cap the bounds
                extended_bounds[0, :3][extended_bounds[0, :3] < -BOUND_CAP] = -BOUND_CAP
                extended_bounds[1, :3][extended_bounds[1, :3] > BOUND_CAP] = BOUND_CAP
                extended_bounds[0, 3:][extended_bounds[0, 3:] < -1.0] = -1.0
                extended_bounds[1, 3:][extended_bounds[1, 3:] > 1.0] = 1.0

                # Sample
                output = np.zeros(7)
                output[:3] = np.random.uniform(
                    extended_bounds[0, :3], extended_bounds[1, :3]
                )
                for ii in range(3, 6):
                    magnitude_used = np.sum(np.square(output[3:ii])) if ii > 3 else 0.0
                    magnitude_left = np.sqrt(1.0 - magnitude_used)
                    min_orient_value = np.max((extended_bounds[0, ii], -magnitude_left))
                    max_orient_value = np.min((extended_bounds[1, ii], magnitude_left))
                    if min_orient_value > max_orient_value:
                        # Sampling orientation failed
                        found_solution = False
                        break
                    output[ii] = np.random.uniform(min_orient_value, max_orient_value)
                output[6] = np.sqrt(1.0 - np.sum(np.square(output[3:6])))
                output = torch.Tensor(output)
            if not found_solution:
                # Return nominal position
                output = torch.Tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
            outputs[i] = output

        norms = torch.linalg.norm(outputs[:, :, 3:], dim=2, keepdim=True) + 1e-8
        norms = norms.reshape(batch.num_graphs, -1, 1).expand(batch.num_graphs, -1, 4)
        normalized_quat = torch.div(outputs[:, :, 3:], norms)

        if self.config.feature_type == "manual":
            oabb = dense_batch[:, 1, 13:19].unsqueeze(1)
            aabb = data_utils.compute_aabb_torch(
                oabb, outputs[:, :, :3], normalized_quat, "cpu"
            )
            outputs = torch.cat((outputs[:, :, :3], aabb, normalized_quat, oabb), dim=2)
        else:
            outputs = torch.cat(
                (
                    outputs[:, :, :3],
                    normalized_quat,
                    dense_batch[:, 1, 7:].unsqueeze(1),
                ),
                dim=2,
            )

        batch.x[batch.ptr[:-1] + 1, :] = outputs[:, 0, :]

        return batch
