import torch
from torch import nn
import torch_geometric as pyg
from highlevel_planning_py.predicate_learning import training_utils as tu
from highlevel_planning_py.predicate_learning import model_utils as mu


class GeneratorMLPPCEncodings(nn.Module):
    def __init__(self, config: tu.MLPNetworkConfig):
        super().__init__()

        self.config = config
        self.out_idx_to_generate = config.custom["out_indices_to_generate"]
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

    def forward(self, input_scene, input_noise=None):
        # Input: [num_scenes, num_features]
        batch_size = input_scene.size(0)
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

        # Assemble output
        output = torch.cat((output[:, :, :3], normalized_quat), dim=2)

        if (
            "out_complete_scene" in self.config.custom
            and self.config.custom["out_complete_scene"]
        ):
            output_scene = input_scene
            num_generated_features = self.config.dim_out / len(
                self.config.custom["out_indices_to_generate"]
            )
            for i, i_arg in enumerate(self.config.custom["out_indices_to_generate"]):
                arg_indices = i_arg + output_scene.ptr[:-1]
                output_scene.x[arg_indices, :num_generated_features] = torch.squeeze(
                    output[:, i, :]
                )

        return output


class GANgeneratorHybridPCEncodings(nn.Module):
    def __init__(self, config: tu.HybridNetworkConfig):
        super().__init__()

        self.config = config

        # Feature inflation layer
        if self.config.inflate_features[1] > self.config.inflate_features[0]:
            self.inflate = nn.Linear(
                self.config.inflate_features[1] - self.config.inflate_features[0],
                self.config.inflate_to,
            )
        else:
            self.inflate = None

        if len(config.config_encoder.layers) > 0:
            self.encode_surrounding = mu.init_encoder(config)

        self.linear_unit = GeneratorMLPPCEncodings(config.config_main_net)

        self.apply(pyg.graphgym.init_weights)

    def forward(self, batch_in, noise, desired_label=None):
        if desired_label is not None:
            raise NotImplementedError

        # Inflate features
        batch = batch_in.clone()
        if self.inflate is not None:
            inflated = self.inflate.forward(
                batch.x[
                    :, self.config.inflate_features[0] : self.config.inflate_features[1]
                ]
            )
            batch.x = torch.cat(
                (
                    batch.x[:, : self.config.inflate_features[0]],
                    inflated,
                    batch.x[:, self.config.inflate_features[1] :],
                ),
                dim=1,
            )

        # Extract argument object features
        argument_features, other_features = mu.extract_feature_tensors(
            batch, self.config.num_argument_objects
        )

        # Assemble input to MLP
        batch_size = batch.num_graphs
        if len(self.config.config_encoder.layers) > 0:
            if self.config.encoder_type == "mlp":
                scene_embedding = self.encode_surrounding.forward(
                    argument_features, other_features
                )
            else:
                scene_embedding = self.encode_surrounding.forward(
                    batch, clone_batch=True
                )
            argument_features = torch.stack(argument_features)
            stacked_features = torch.cat(
                (argument_features.reshape((batch_size, -1)), scene_embedding, noise),
                dim=1,
            )
        else:
            argument_features = torch.stack(argument_features)
            stacked_features = torch.cat(
                (argument_features.reshape((batch_size, -1)), noise), dim=1
            )

        # MLP
        new_features = self.linear_unit.forward(stacked_features)

        # Assemble output
        batch_out = batch_in.clone()
        for i, i_arg in enumerate(self.config.custom["out_indices_to_generate"]):
            arg_indices = i_arg + batch_out.ptr[:-1]
            arg_indices = arg_indices.tolist()
            batch_out.x[
                arg_indices, : self.config.config_main_net.dim_out
            ] = new_features[:, i, :]

        return batch_out

    @staticmethod
    def sample_noise(num_samples, dim_noise):
        return torch.randn((num_samples, dim_noise))
