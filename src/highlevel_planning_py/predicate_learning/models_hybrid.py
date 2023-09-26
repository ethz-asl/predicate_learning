import torch
from torch import nn
import torch_geometric as pyg
from highlevel_planning_py.predicate_learning import training_utils
from highlevel_planning_py.predicate_learning.models_gnn import GenerationGNN
from highlevel_planning_py.predicate_learning.models_mlp import (
    DiscriminatorMLP,
    GeneratorMLP,
)
from highlevel_planning_py.predicate_learning import model_utils as mu


class GANdiscriminatorHybrid(nn.Module):
    def __init__(self, config: training_utils.HybridNetworkConfig):
        super().__init__()
        self.config = config

        # Feature inflation layer
        if (
            hasattr(self.config, "inflate_features")
            and len(self.config.inflate_features) == 2
            and self.config.inflate_features[1] > self.config.inflate_features[0]
        ):
            self.inflate = nn.Linear(
                self.config.inflate_features[1] - self.config.inflate_features[0],
                self.config.inflate_to,
            )
        else:
            self.inflate = None

        # Scene encoder
        if len(self.config.config_encoder.layers) > 0:
            self.encode_surrounding = mu.init_encoder(config)

        # MLP component
        self.discriminate = DiscriminatorMLP(config.config_main_net)

        self.apply(pyg.graphgym.init_weights)

    def forward(self, batch_in, predicate_states=None):
        batch = batch_in.clone()

        # Continue with only selected features
        if "in_feature_selector" in self.config.custom:
            batch.x = batch.x[:, self.config.custom["in_feature_selector"]]

        # Inflate features
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

        if hasattr(batch, "num_graphs"):
            batch_size = batch.num_graphs
        else:
            batch_size = 1
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
                (argument_features.reshape((batch_size, -1)), scene_embedding), dim=1
            )
        else:
            argument_features = torch.stack(argument_features)
            stacked_features = argument_features.reshape((batch_size, -1))

        if predicate_states is not None:
            stacked_features = torch.cat((stacked_features, predicate_states), dim=1)

        prediction = self.discriminate.forward(stacked_features)
        return prediction


class GANgeneratorHybridV1(nn.Module):
    def __init__(self, config: training_utils.HybridNetworkConfig):
        super().__init__()
        self.config = config

        assert config.config_encoder.post_processing is False
        self.message_passing = GenerationGNN(config.config_encoder)

        self.argument_units = list()
        for i in range(config.num_argument_objects):
            self.argument_units.append(GeneratorMLP(config.config_main_net))

        self.surrounding_unit = GeneratorMLP(config.config_main_net)

        self.apply(pyg.graphgym.init_weights)

    def forward(self, batch_in, noise, desired_label=None):
        batch = batch_in.clone()

        if desired_label is not None:
            raise NotImplementedError

        # Select features
        batch.x = batch.x[:, self.config.custom["in_feature_selector"]]

        output = self.message_passing.forward(batch)
        new_features = torch.clone(batch.x)

        # Predict argument objects
        for i_arg in self.config.custom["out_indices_to_generate"]:
            arg_indices = i_arg + output.ptr[:-1]
            new_features[arg_indices, :] = torch.squeeze(
                self.argument_units[i_arg].forward(
                    torch.unsqueeze(output.x[arg_indices, :], dim=1),
                    oabb=torch.unsqueeze(batch.x[arg_indices, 13:19], dim=1),
                    input_noise=noise,
                ),
                dim=1,
            )

        # Predict surrounding objects
        if -1 in self.config.custom["out_indices_to_generate"]:
            surrounding_indices = list()
            for i in range(output.num_graphs):
                if output.ptr[i] + self.config.num_argument_objects < output.ptr[i + 1]:
                    new_indices = torch.arange(
                        output.ptr[i] + self.config.num_argument_objects,
                        output.ptr[i + 1],
                    )
                    surrounding_indices.append(new_indices)
            if len(surrounding_indices) > 0:
                surrounding_indices = torch.cat(surrounding_indices)
                new_features[surrounding_indices, :] = torch.squeeze(
                    self.surrounding_unit.forward(
                        torch.unsqueeze(output.x[surrounding_indices, :], dim=1),
                        oabb=torch.unsqueeze(
                            batch.x[surrounding_indices, 13:19], dim=1
                        ),
                        input_noise=noise,
                    ),
                    dim=1,
                )

        output.x = new_features
        return output

    @staticmethod
    def sample_noise(num_samples, dim_noise):
        return torch.randn((num_samples, dim_noise))


class GANgeneratorHybridV2(nn.Module):
    def __init__(self, config: training_utils.HybridNetworkConfig):
        super().__init__()
        self.config = config

        if len(config.config_encoder.layers) > 0:
            self.encode_surrounding = mu.init_encoder(config)

        self.linear_unit = GeneratorMLP(config.config_main_net)

        self.apply(pyg.graphgym.init_weights)

    def forward(self, batch_in, noise, desired_label=None):
        batch = batch_in.clone()

        # Select features
        if "in_feature_selector" in self.config.custom:
            batch.x = batch.x[:, self.config.custom["in_feature_selector"]]

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

        if desired_label is not None:
            stacked_features = torch.cat((stacked_features, desired_label), dim=1)

        # MLP
        output = self.linear_unit.forward(
            stacked_features,
            oabb=argument_features[
                :, self.config.custom["out_indices_to_generate"], 13:19
            ],
            assemble_output=False,
        )

        # Assemble output
        for i, i_arg in enumerate(self.config.custom["out_indices_to_generate"]):
            arg_indices = i_arg + batch.ptr[:-1]
            batch.x[arg_indices, :] = torch.squeeze(output[:, i, :])

        return batch

    @staticmethod
    def sample_noise(num_samples, dim_noise):
        return torch.randn((num_samples, dim_noise))
