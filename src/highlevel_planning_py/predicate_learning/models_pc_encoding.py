import torch
from torch import nn
import torch_geometric as pyg
import torch.nn.functional as F
from torch_geometric.nn import (
    MLP,
    PointConv,
    fps,
    global_max_pool,
    radius,
    knn_interpolate,
)

# Adapted from https://github.com/pyg-team/pytorch_geometric/blob/master/examples/pointnet2_classification.py


class SAModule(torch.nn.Module):
    def __init__(self, ratio, r, nn):
        super().__init__()
        self.ratio = ratio
        self.r = r
        self.conv = PointConv(nn, add_self_loops=False)

    def forward(self, x, pos, batch):
        idx = fps(pos, batch, ratio=self.ratio)
        row, col = radius(
            pos, pos[idx], self.r, batch, batch[idx], max_num_neighbors=64
        )
        edge_index = torch.stack([col, row], dim=0)
        x_dst = None if x is None else x[idx]
        x = self.conv((x, x_dst), (pos, pos[idx]), edge_index)
        pos, batch = pos[idx], batch[idx]
        return x, pos, batch


class GlobalSAModule(torch.nn.Module):
    def __init__(self, nn):
        super().__init__()
        self.nn = nn

    def forward(self, x, pos, batch):
        x = self.nn(torch.cat([x, pos], dim=1))
        x = global_max_pool(x, batch)
        pos = pos.new_zeros((x.size(0), 3))
        batch = torch.arange(x.size(0), device=batch.device)
        return x, pos, batch


class Pointnet2Encoder(torch.nn.Module):
    def __init__(self, encoding_dim: int, model_config: str, autoencoder=False):
        super().__init__()

        self.autoencoder = autoencoder

        # Input channels account for both `pos` and node features.
        if model_config == "original":
            self.sa1_module = SAModule(0.5, 0.2, MLP([3, 64, 64, 128]))
            self.sa2_module = SAModule(0.25, 0.4, MLP([128 + 3, 128, 128, 256]))
            self.sa3_module = GlobalSAModule(MLP([256 + 3, 256, 512, encoding_dim]))
        elif model_config == "small":
            self.sa1_module = SAModule(0.2, 0.2, MLP([3, 32, 64]))
            self.sa2_module = SAModule(0.25, 0.4, MLP([64 + 3, 64, 128]))
            self.sa3_module = GlobalSAModule(MLP([128 + 3, 128, encoding_dim]))
        else:
            raise ValueError

        # self.mlp = MLP([1024, 512, 256, 10], dropout=0.5, norm=None)

    def forward(self, data):
        sa0_out = (data.x, data.pos, data.batch)
        sa1_out = self.sa1_module(*sa0_out)
        sa2_out = self.sa2_module(*sa1_out)
        sa3_out = self.sa3_module(*sa2_out)
        # x, pos, batch = sa3_out

        # return self.mlp(x).log_softmax(dim=-1)

        if self.autoencoder:
            return sa0_out, sa1_out, sa2_out, sa3_out
        else:
            return sa3_out[0]


class FPModule(torch.nn.Module):
    def __init__(self, k, nn):
        super().__init__()
        self.k = k
        self.nn = nn

    def forward(self, x, pos, batch, x_skip, pos_skip, batch_skip):
        x = knn_interpolate(x, pos, pos_skip, batch, batch_skip, k=self.k)
        if x_skip is not None:
            x = torch.cat([x, x_skip], dim=1)
        x = self.nn(x)
        return x, pos_skip, batch_skip


class Pointnet2Decoder(torch.nn.Module):
    def __init__(self, encoding_dim: int, model_config: str):
        super(Pointnet2Decoder, self).__init__()

        self.nonlin = F.relu

        if model_config == "original":
            self.fp3_module = FPModule(1, MLP([encoding_dim + 256, 256, 256]))
            self.fp2_module = FPModule(3, MLP([256 + 128, 256, 128]))
            self.fp1_module = FPModule(3, MLP([128, 128, 128, 64]))
            self.lin = nn.ModuleList(
                [
                    torch.nn.Linear(64, 32),
                    torch.nn.Linear(32, 32),
                    torch.nn.Linear(32, 3),
                ]
            )
        elif model_config == "small":
            self.fp3_module = FPModule(1, MLP([encoding_dim + 128, 256, 128]))
            self.fp2_module = FPModule(3, MLP([128 + 64, 128, 64]))
            self.fp1_module = FPModule(3, MLP([64, 64, 64]))
            self.lin = nn.ModuleList(
                [
                    torch.nn.Linear(64, 32),
                    torch.nn.Linear(32, 32),
                    torch.nn.Linear(32, 3),
                ]
            )
        else:
            raise ValueError

    def forward(self, data):
        fp3_out = self.fp3_module(*data[3], *data[2])
        fp2_out = self.fp2_module(*fp3_out, *data[1])
        x, pos_x, _ = self.fp1_module(*fp2_out, *data[0])

        for lin_layer in self.lin[:-1]:
            x = lin_layer.forward(x)
            x = self.nonlin(x)
        x = self.lin[-1].forward(x)

        return x


# ----- The following Convolutional models are adapted from Achlioptas' representation learning models
# (https://github.com/optas/latent_3d_points)
# And its pytorch implementation (https://github.com/TropComplique/point-cloud-autoencoder)


class ConvEncoder(nn.Module):
    def __init__(self, enc_dim, batch_norm=False, model_config="original"):
        super(ConvEncoder, self).__init__()

        pointwise_layers = list()
        if model_config == "original":
            layer_spec = [3, 64, 128, 128, 256, enc_dim]
        elif model_config == "small":
            layer_spec = [3, 128, 256, enc_dim]
        elif model_config == "large":
            layer_spec = [3, 64, 128, 256, 256, 512, enc_dim]
        else:
            raise ValueError
        for ch_in, ch_out in zip(layer_spec[:-1], layer_spec[1:]):
            pointwise_layers.append(nn.Conv1d(ch_in, ch_out, kernel_size=1, bias=False))
            if batch_norm:
                pointwise_layers.append(nn.BatchNorm1d(ch_out))
            pointwise_layers.append(nn.ReLU())
        self.pointwise_layers = nn.Sequential(*pointwise_layers)
        self.pooling = nn.AdaptiveMaxPool1d(1)

    def forward(self, data):
        pos_batched, pos_mask = pyg.utils.to_dense_batch(data.pos, data.batch)
        pos_batched = torch.transpose(pos_batched, 1, 2)
        pointwise_enc = self.pointwise_layers(pos_batched)
        global_enc = self.pooling(pointwise_enc)
        global_enc = torch.squeeze(global_enc, dim=2)
        return global_enc


class ConvDecoder(nn.Module):
    def __init__(self, enc_dim, num_points, batch_norm=False, model_config="original"):
        super(ConvDecoder, self).__init__()

        layers = list()
        if model_config == "original":
            layer_spec = [enc_dim, 256, 256, num_points * 3]
        elif model_config == "small":
            layer_spec = [enc_dim, 512, num_points * 3]
        elif model_config == "large":
            layer_spec = [enc_dim, 256, 512, 1024, num_points * 3]
        else:
            raise ValueError
        for ch_in, ch_out in zip(layer_spec[:-2], layer_spec[1:-1]):
            layers.append(nn.Conv1d(ch_in, ch_out, kernel_size=1, bias=False))
            if batch_norm:
                layers.append(nn.BatchNorm1d(ch_out))
            layers.append(nn.ReLU())
        layers.append(nn.Conv1d(layer_spec[-2], layer_spec[-1], kernel_size=1))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        # x: tensor with shape [bs, enc_dim]
        # bs, enc_dim = x.size()

        x_reshaped = torch.unsqueeze(x, 2)

        points = self.layers(x_reshaped)
        points = points.view(-1, 3)

        return points


class FullyConnectedDecoder(nn.Module):
    def __init__(self, enc_dim, num_points, model_config="original"):
        super(FullyConnectedDecoder, self).__init__()

        layers = list()
        if model_config == "original":
            layer_spec = [enc_dim, 256, 512, num_points * 3]
        elif model_config == "small":
            layer_spec = [enc_dim, 512, num_points * 3]
        elif model_config == "large":
            layer_spec = [enc_dim, 256, 512, 1024, num_points * 3]
        else:
            raise ValueError
        for len_in, len_out in zip(layer_spec[:-2], layer_spec[1:-1]):
            layers.append(nn.Linear(len_in, len_out, bias=False))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(layer_spec[-2], layer_spec[-1], bias=True))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        # x: tensor with shape [bs, enc_dim]

        points = self.layers(x)
        points = points.view(-1, 3)
        return points
