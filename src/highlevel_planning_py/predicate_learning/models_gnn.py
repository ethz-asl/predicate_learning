import torch_geometric as pyg
import torch
from torch import nn
from highlevel_planning_py.predicate_learning import data_utils, training_utils
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool


class WeightClipperClamp:
    def __init__(self, clamp_min, clamp_max):
        self.min = clamp_min
        self.max = clamp_max

    def __call__(self, module, *args, **kwargs):
        if hasattr(module, "weight"):
            w = module.weight.data
            torch.clamp(w, self.min, self.max, out=w)


class WeightClipperNorm:
    def __init__(self, radius=1.0):
        self.radius = radius

    def __call__(self, module, *args, **kwargs):
        if hasattr(module, "weight"):
            w = module.weight.data
            if w.ndim > 1:
                w.div_(torch.norm(w, 2, 1).view(-1, 1).expand_as(w))
            else:
                w.div_(torch.norm(w, 2).expand_as(w))
            w.mul_(self.radius)


def new_layer_config(dim_in, dim_out, dim_inner, num_layers, has_act, has_bias):
    return pyg.graphgym.models.layer.LayerConfig(
        has_batchnorm=False,
        bn_eps=1e-5,
        bn_mom=0.1,
        mem_inplace=False,
        dim_in=dim_in,
        dim_out=dim_out,
        num_layers=num_layers,
        has_l2norm=False,
        dropout=0.0,
        has_act=has_act,
        final_act=True,
        act="prelu",
        has_bias=has_bias,
        dim_inner=dim_inner,
    )


def create_mp_layer(layer_type, dim_in, dim_out, dim_inner, has_act=True):
    layer_config = new_layer_config(
        dim_in, dim_out, dim_inner, num_layers=1, has_act=has_act, has_bias=False
    )
    # return pyg.graphgym.models.layer.GeneralLayer(layer_type, layer_config)
    return pyg.graphgym.models.layer.GeneralLayer(layer_type, layer_config)


class GraphHead(nn.Module):
    def __init__(self, dim_in, dim_out, dim_inner, num_layers, pooling_type):
        super(GraphHead, self).__init__()
        if pooling_type == "add":
            self.pooling = global_add_pool
        elif pooling_type == "mean":
            self.pooling = global_mean_pool
        elif pooling_type == "max":
            self.pooling = global_max_pool
        else:
            raise ValueError("Invalid pooling type")

        if num_layers > 0:
            layer_config = new_layer_config(
                dim_in, dim_out, dim_inner, num_layers, has_act=False, has_bias=True
            )
            self.layer_post_mp = pyg.graphgym.models.layer.MLP(layer_config)
        else:
            self.layer_post_mp = None

    def forward(self, batch):
        graph_emb = self.pooling(batch.x, batch.batch)
        if self.layer_post_mp is not None:
            graph_emb = self.layer_post_mp(graph_emb)
        batch.graph_feature = graph_emb
        return graph_emb


class ClassificationGNN(nn.Module):
    def __init__(self, config: training_utils.GNNNetworkConfig):
        super().__init__()

        for i in range(config.layers[1]):
            dim_in = config.dim_in if i == 0 else config.dim_inner[1]
            layer = create_mp_layer(
                config.mp_layer_type, dim_in, config.dim_inner[1], config.dim_inner[1]
            )
            self.add_module(f"mp_layer{i}", layer)
        layer = GraphHead(
            config.dim_inner[1],
            config.dim_out,
            config.dim_inner[2],
            config.layers[2],
            config.graph_pooling,
        )
        self.add_module("head", layer)

        self.apply(pyg.graphgym.init_weights)

    def forward(self, batch):
        for module in self.children():
            batch = module(batch)
        return batch


class GenerationGNN(nn.Module):
    def __init__(self, config: training_utils.GNNNetworkConfig):
        super().__init__()
        self.quat_idx_out = config.custom["quat_idx_out"]
        self.device = config.device
        self.post_processing = config.post_processing
        self.eps = config.eps if hasattr(config, "eps") else 1e-8

        dim_in = config.dim_in
        # Pre MP
        if config.layers[0] > 0:
            layer_config = new_layer_config(
                dim_in,
                config.dim_inner[0],
                config.dim_inner[0],
                config.layers[0],
                has_act=False,
                has_bias=False,
            )
            layer = pyg.graphgym.models.layer.GeneralMultiLayer("linear", layer_config)
            self.add_module("pre_mp_layer", layer)
            dim_in = config.dim_inner[0]
        # MP
        if config.layers[1] > 0:
            for i in range(config.layers[1]):
                dim_in = dim_in if i == 0 else config.dim_inner[1]
                d_out = (
                    config.dim_out
                    if i == config.layers[1] - 1 and config.layers[2] == 0
                    else config.dim_inner[1]
                )
                layer = create_mp_layer(
                    config.mp_layer_type, dim_in, d_out, config.dim_inner[1]
                )
                self.add_module(f"mp_layer{i}", layer)
            dim_in = config.dim_inner[1]
        # Post MP
        if config.layers[2] > 0:
            layer_config = new_layer_config(
                dim_in,
                config.dim_out,
                config.dim_inner[2],
                config.layers[2],
                has_act=False,
                has_bias=True,
            )
            layer = pyg.graphgym.models.layer.MLP(layer_config)
            self.add_module("post_mp_layer", layer)
        self.apply(pyg.graphgym.init_weights)

    def forward(self, batch, noise=None):
        output = batch.clone()

        if noise is not None:
            propagated_noise = propagate_over_batches(batch, noise, self.device)
            output.x = torch.cat((output.x, propagated_noise), dim=1)
        for mod in self.children():
            output = mod(output)

        if self.post_processing:
            # Force normalized quaternion
            norms = (
                torch.linalg.norm(
                    output.x[:, self.quat_idx_out : self.quat_idx_out + 4], dim=1
                )
                + self.eps
            )
            norms = norms.reshape(-1, 1).expand(output.x.shape[0], 4)
            normalized_quat = torch.div(
                output.x[:, self.quat_idx_out : self.quat_idx_out + 4], norms
            )

            # Compute aabb
            aabb = data_utils.compute_aabb_torch(
                batch.x[:, 13:19].unsqueeze(0),
                output.x[:, :3].unsqueeze(0),
                normalized_quat.unsqueeze(0),
                self.device,
            )
            aabb = torch.squeeze(aabb)

            # Form output
            output.x = torch.cat(
                (output.x[:, :3], aabb, normalized_quat, batch.x[:, 13:19]), dim=1
            )
        return output


def propagate_over_batches(batch, states, device):
    if states.ndim == 1:
        cols = 1
    else:
        cols = states.size(-1)
    propagated_predicate_states = torch.zeros(batch.num_nodes, cols).to(device)
    prev_idx = 0
    for i in range(1, len(batch._slice_dict["x"])):
        num_nodes_this_sample = batch._slice_dict["x"][i] - prev_idx
        propagated_predicate_states[prev_idx : batch._slice_dict["x"][i], :] = states[
            i - 1
        ].expand(num_nodes_this_sample.item(), cols)
        prev_idx = batch._slice_dict["x"][i].item()
    return propagated_predicate_states


class GANdiscriminatorGNN(nn.Module):
    def __init__(self, config: training_utils.GNNNetworkConfig):
        super().__init__()
        self.device = config.device

        dim_in = config.dim_in
        # Pre MP
        if config.layers[0] > 0:
            layer_config = pyg.graphgym.models.layer.LayerConfig(
                dim_in=dim_in,
                dim_out=config.dim_inner[0],
                dim_inner=config.dim_inner[0],
                num_layers=config.layers[0],
            )
            layer = pyg.graphgym.models.layer.GeneralMultiLayer("linear", layer_config)
            self.add_module("pre_mp_layers", layer)
            dim_in = config.dim_inner[0]
        # MP
        for i in range(config.layers[1]):
            dim_in = dim_in if i == 0 else config.dim_inner[1]
            layer = create_mp_layer(
                config.mp_layer_type, dim_in, config.dim_inner[1], config.dim_inner[1]
            )
            self.add_module(f"mp_layer{i}", layer)
            if i == config.layers[1] - 1:
                dim_in = config.dim_inner[1]
        # Post MP
        layer = GraphHead(
            dim_in,
            config.dim_out,
            config.dim_inner[2],
            config.layers[2],
            config.graph_pooling,
        )
        self.add_module("head", layer)

        self.apply(pyg.graphgym.init_weights)

    def forward(self, batch, predicate_states=None, clone_batch=True):
        if clone_batch:
            output = batch.clone()
        else:
            output = batch

        # Append graph labels to each node
        if predicate_states is not None:
            propagated_predicate_states = propagate_over_batches(
                batch, predicate_states, self.device
            )
            output.x = torch.cat((output.x, propagated_predicate_states), dim=1)

        # Run GNN
        for module in self.children():
            output = module(output)
        return output


class GANgeneratorGNN(nn.Module):
    def __init__(self, config: training_utils.GNNNetworkConfig):
        super().__init__()
        self.device = config.device
        self.graph_generator = GenerationGNN(config)

    def forward(self, batch, noise, desired_label=None, clone_batch=True):
        if clone_batch:
            output = batch.clone()
        else:
            output = batch

        # Append desired graph labels to each node
        if desired_label is not None:
            propagated_predicate_states = propagate_over_batches(
                output, desired_label, self.device
            )
            output.x = torch.cat((output.x, propagated_predicate_states), dim=1)

        # Run GNN
        output = self.graph_generator.forward(output, noise)
        if desired_label is not None:
            output.y = desired_label.squeeze()

        return output

    @staticmethod
    def sample_noise(num_samples, dim_noise):
        return torch.randn((num_samples, dim_noise))
