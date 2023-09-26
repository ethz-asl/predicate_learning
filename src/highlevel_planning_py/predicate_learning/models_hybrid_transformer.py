from torch import nn

import highlevel_planning_py.predicate_learning.training_utils as tu
from highlevel_planning_py.predicate_learning.models_mlp import DiscriminatorMLP


class DiscriminatorHybridTransformer(nn.Module):
    def __init__(self, config: tu.HybridTransformerNetworkConfig):
        super().__init__()
        self.config = config

        # Scene encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.num_features,
            nhead=config.num_heads,
            dim_feedforward=config.dim_feedforward,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=config.num_encoder_layers
        )

        # Classifier MLP
        self.discriminate = DiscriminatorMLP(config.config_mlp)

    def forward(self, batch_in, predicate_states=None):

        batch_size = batch_in.size(0)

        if "in_feature_selector" in self.config.custom:
            selected_input_features = batch_in[
                :, :, self.config.custom["in_feature_selector"]
            ]
        else:
            selected_input_features = batch_in

        # scene_encoding =

        # WIP
