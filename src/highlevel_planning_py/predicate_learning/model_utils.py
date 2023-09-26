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

from typing import Tuple, List, Callable
import torch
from torch import nn
import torch_geometric as pyg
from highlevel_planning_py.predicate_learning import training_utils
from highlevel_planning_py.predicate_learning.models_gnn import GANdiscriminatorGNN


def extract_feature_tensors(batch, num_argument_objects) -> Tuple[List, List]:
    if hasattr(batch, "num_graphs"):
        batch_size = batch.num_graphs
    else:
        batch_size = 1
    argument_features = list()
    other_features = list()
    for i in range(batch_size):
        if batch_size > 1:
            single_scene = batch.get_example(i)
        else:
            single_scene = batch
        argument_features.append(single_scene.x[:num_argument_objects, :])
        other_features.append(single_scene.x[num_argument_objects:, :])
    return argument_features, other_features


def setup_mlp_encoder(config_encoder: training_utils.MLPNetworkConfig):
    # MLP
    linear_layers = pyg.nn.MLP(
        [config_encoder.dim_in, *config_encoder.layers, config_encoder.dim_out],
        norm=None,
    )

    # Pooling
    if config_encoder.custom["graph_pooling"] == "add":
        pooling = torch.sum
    elif config_encoder.custom["graph_pooling"] == "mean":
        pooling = torch.mean
    # elif config.custom["graph_pooling"] == "max":
    #     self.pooling = pyg.nn.global_max_pool
    else:
        raise ValueError("Invalid pooling type")

    return linear_layers, pooling


def encode_scene(
    encoder: nn.Module,
    pooling: Callable,
    argument_features: List[torch.tensor],
    other_features: List[torch.tensor],
    batch_size: int,
    encoder_dim_in: int,
    device: str,
) -> List[torch.tensor]:
    encoder_inputs = list()
    batch_sep = [0]
    for i in range(batch_size):
        # Assemble encoder input
        if other_features[i].numel() > 0:
            arg_features = (
                argument_features[i]
                .reshape((1, -1))
                .expand(other_features[i].size(0), -1)
            )
            encoder_input = torch.cat((arg_features, other_features[i]), dim=1)
            batch_sep.append(batch_sep[-1] + encoder_input.size(0))
            encoder_inputs.append(encoder_input)
        else:
            encoder_input = torch.zeros((1, encoder_dim_in)).to(device)
            batch_sep.append(batch_sep[-1] + 1)
            encoder_inputs.append(encoder_input)
    encoder_inputs_combined = torch.cat(encoder_inputs, dim=0)
    encoding = encoder.forward(encoder_inputs_combined)
    encodings_separated = [
        encoding[batch_sep[i] : batch_sep[i + 1], :] for i in range(batch_size)
    ]
    encodings_separated = [
        pooling(encodings_separated[i], dim=0) for i in range(len(encodings_separated))
    ]
    return encodings_separated


class GraphEncoder(nn.Module):
    """
    This is a GNN encoder that is based on torch_geometric functions, not using graphgym
    implementations.
    """

    def __init__(self, config: training_utils.GNNNetworkConfig):
        super().__init__()

        # Message passing
        self.mp = pyg.nn.GAT(
            in_channels=config.dim_in,
            hidden_channels=config.dim_inner[1],
            num_layers=config.layers[1],
            out_channels=config.dim_out,
            jk=config.custom["jk"] if "jk" in config.custom else None,
        )

        # Pooling
        if config.graph_pooling == "add":
            self.pooling = pyg.nn.global_add_pool
        elif config.graph_pooling == "mean":
            self.pooling = pyg.nn.global_mean_pool
        elif config.graph_pooling == "max":
            self.pooling = pyg.nn.global_max_pool
        else:
            raise ValueError("Invalid pooling type")

    def forward(self, batch, clone_batch=True):
        if clone_batch:
            batch_in = batch.clone()
        else:
            batch_in = batch
        batch_in.x = self.mp.forward(batch_in.x, batch_in.edge_index)
        graph_emb = self.pooling(batch_in.x, batch_in.batch)
        return graph_emb


class ManualMLPEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.encoder_layers, self.pooling = setup_mlp_encoder(config)

    def forward(
        self, argument_features: List[torch.tensor], other_features: List[torch.tensor]
    ):
        batch_size = len(argument_features)
        encodings = encode_scene(
            self.encoder_layers,
            self.pooling,
            argument_features,
            other_features,
            batch_size,
            self.config.dim_in,
            self.config.device,
        )
        encodings = torch.stack(encodings)
        return encodings


def init_encoder(config: training_utils.HybridNetworkConfig):
    if config.encoder_type == "own":
        assert config.config_encoder.post_processing is False
        assert (
            config.config_encoder.layers[2] > 0
        ), "Need to have at least one post MP layer to assure correct output size."
        scene_encoder = GANdiscriminatorGNN(config.config_encoder)
    elif config.encoder_type == "gat":
        scene_encoder = GraphEncoder(config.config_encoder)
    elif config.encoder_type == "mlp":
        scene_encoder = ManualMLPEncoder(config.config_encoder)
    else:
        raise NotImplementedError("Unknown encoder GNN type")
    return scene_encoder
