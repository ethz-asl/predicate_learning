import torch
import torch.nn as nn
import torch_geometric as pyg

from highlevel_planning_py.predicate_learning import training_utils as tu
from highlevel_planning_py.predicate_learning import data_utils


class GeneratorMLP(nn.Module):
    def __init__(self, config: tu.MLPNetworkConfig):
        super().__init__()
        self.out_idx_to_generate = config.custom["out_indices_to_generate"]
        self.config = config

        activation_function = tu.get_activation_function(config.activation_type)

        module_list = list()
        # First layers
        dim_last = -1
        for i_layer, num_neurons in enumerate(config.layers):
            dim_in = config.dim_in if i_layer == 0 else dim_last
            dim_last = num_neurons
            module_list.append(nn.Linear(dim_in, num_neurons))
            module_list.append(activation_function())
        # Final layer
        dim_in = config.dim_in if len(module_list) == 0 else dim_last
        module_list.append(nn.Linear(dim_in, config.dim_out))
        self.latent_to_scene = nn.Sequential(*module_list)

        self.apply(pyg.graphgym.init_weights)

    def forward(
        self,
        input_scene,
        input_noise=None,
        oabb=None,
        desired_label=None,
        assemble_output=True,
    ):
        if desired_label is not None:
            raise NotImplementedError

        # Input: [batch_size, num_objects, dim_in]
        batch_size = input_scene.size(0)
        if "in_feature_selector" in self.config.custom:
            selected_input_features = input_scene[
                :, :, self.config.custom["in_feature_selector"]
            ]
        else:
            selected_input_features = input_scene
        input_scene_reshaped = torch.reshape(selected_input_features, (batch_size, -1))
        if input_noise is not None:
            input_cat = torch.cat((input_noise, input_scene_reshaped), dim=1)
        else:
            input_cat = input_scene_reshaped
        output = self.latent_to_scene(input_cat)
        output = torch.reshape(output, (batch_size, len(self.out_idx_to_generate), -1))

        # Force normalized quaternion
        quat_start_idx = self.config.custom["quat_idx_out"]
        eps = self.config.eps if hasattr(self.config, "eps") else 1e-8
        norms = (
            torch.linalg.norm(output[:, :, quat_start_idx : quat_start_idx + 4], dim=2)
            + eps
        )
        norms = norms.reshape(batch_size, -1, 1).expand(batch_size, -1, 4)
        normalized_quat = torch.div(
            output[:, :, quat_start_idx : quat_start_idx + 4], norms
        )

        if oabb is None:
            idx_in_oabb = self.config.custom["idx_in_oabb"]
            oabb = input_scene[
                :, self.out_idx_to_generate, idx_in_oabb : idx_in_oabb + 6
            ]

        if self.config.feature_version == "v1":
            # Compute aabb
            aabb = data_utils.compute_aabb_torch(
                oabb, output[:, :, :3], normalized_quat, self.config.device
            )

            output = torch.cat((output[:, :, :3], aabb, normalized_quat, oabb), dim=2)
        elif self.config.feature_version == "v2":
            shifted_oabb = data_utils.shift_oabb(
                oabb, output[:, :, :3], normalized_quat, self.config.device
            )
            shifted_oabb = shifted_oabb[:, :, [0, 1, 2, 4], :]
            shifted_oabb = torch.reshape(shifted_oabb, (batch_size, -1, 12))
            output = torch.cat(
                (output[:, :, :3], normalized_quat, shifted_oabb, oabb), dim=2
            )
        elif self.config.feature_version == "v3":
            shifted_oabb = data_utils.shift_oabb(
                oabb, output[:, :, :3], normalized_quat, self.config.device
            )
            oabb_surface_corners = torch.stack(
                (
                    shifted_oabb[:, :, 4:, :],
                    shifted_oabb[:, :, [0, 2, 4, 6], :],
                    shifted_oabb[:, :, [0, 1, 4, 5], :],
                ),
                dim=2,
            )
            oabb_surface_centers = torch.mean(oabb_surface_corners, dim=3)
            oabb_surface_centers = torch.reshape(
                oabb_surface_centers, (batch_size, -1, 9)
            )
            output = torch.cat(
                (output[:, :, :3], normalized_quat, oabb_surface_centers, oabb), dim=2
            )
        else:
            raise NotImplementedError("Unknown feature version")

        # Assemble output from generated and constant features
        if assemble_output:
            combined_output = list()
            for i in range(input_scene.size(1)):
                if i in self.out_idx_to_generate:
                    i_output = self.out_idx_to_generate.index(i)
                    combined_output.append(output[:, i_output, :].unsqueeze(1))
                else:
                    combined_output.append(input_scene[:, i, :].unsqueeze(1))
            combined_output = torch.cat(combined_output, dim=1)
        else:
            combined_output = output

        # Re-normalize generated sample
        if self.config.custom["normalize_output"]:
            if self.config.custom["normalization_method"] == "scene_center":
                reference = data_utils.compute_scene_centroid_torch(
                    combined_output, self.config.feature_version
                )
                reference = reference.expand(-1, combined_output.size(dim=1), -1)
            elif self.config.custom["normalization_method"] == "first_arg":
                reference = combined_output[:, 0, :3].unsqueeze(1)
                reference = reference.expand(-1, combined_output.size(dim=1), -1)
            else:
                raise NotImplementedError
            combined_output = data_utils.normalize_features_torch(
                combined_output, reference, self.config.feature_version
            )

        return combined_output

    @staticmethod
    def sample_noise(num_samples, dim_noise):
        return torch.randn((num_samples, dim_noise))


class DiscriminatorMLP(nn.Module):
    def __init__(self, config: tu.MLPNetworkConfig):
        super().__init__()

        self.config = config

        activation_function = tu.get_activation_function(config.activation_type)

        module_list = list()

        # First layers
        dim_last = -1
        for i_layer, num_neurons in enumerate(config.layers):
            dim_in = config.dim_in if i_layer == 0 else dim_last
            dim_last = num_neurons
            module_list.append(nn.Linear(dim_in, num_neurons))
            module_list.append(activation_function())
        # Final layer
        dim_in = config.dim_in if len(module_list) == 0 else dim_last
        module_list.append(nn.Linear(dim_in, config.dim_out))
        self.scene_to_prob = nn.Sequential(*module_list)

        self.apply(pyg.graphgym.init_weights)

    def forward(self, input_data, predicate_label=None):
        if predicate_label is not None:
            raise NotImplementedError
        batch_size = input_data.size(0)
        if "in_feature_selector" in self.config.custom:
            selected_input_features = input_data[
                :, :, self.config.custom["in_feature_selector"]
            ]
        else:
            selected_input_features = input_data
        input_vec = selected_input_features.reshape((batch_size, -1))
        output = self.scene_to_prob(input_vec)
        return output
