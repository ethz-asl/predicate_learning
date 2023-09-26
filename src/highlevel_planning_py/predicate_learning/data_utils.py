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

import numpy as np
import torch
from torch_geometric.data.data import Data as GraphData
from scipy.spatial.transform.rotation import Rotation as R
import itertools
from functools import lru_cache
from pytorch3d import transforms


def normalize_features(features, reference, feature_version):
    """
    Takes reference features vector (e.g. main considered object in this data point)
    and normalizes the other feature vectors using the reference.

    Args:
        features:           np array of shape [n, m] where n is number of feature vectors and
                            m is number of features per feature vector.
        reference:          np array of shape [m] with the reference feature vector.
        feature_version:    Version of features to use. This influences which values are
                            normalized.
    """
    ret = np.copy(features)
    if len(features) > 0:
        ret[:, :3] -= reference[:3]
        if feature_version == "v1":
            ret[:, 3:6] -= reference[:3]
            ret[:, 6:9] -= reference[:3]
        elif feature_version == "v2":
            ret[:, 7:10] -= reference[:3]
            ret[:, 10:13] -= reference[:3]
            ret[:, 13:16] -= reference[:3]
            ret[:, 16:19] -= reference[:3]
        elif feature_version == "v3":
            ret[:, 7:10] -= reference[:3]
            ret[:, 10:13] -= reference[:3]
            ret[:, 13:16] -= reference[:3]
        else:
            raise ValueError
    return ret


def unnormalize_features(features, reference, feature_version):
    ret = np.copy(features)
    if len(features) > 0:
        ret[:, :3] += reference[:3]
        if feature_version == "v1":
            ret[:, 3:6] += reference[:3]
            ret[:, 6:9] += reference[:3]
        elif feature_version == "v2":
            ret[:, 7:10] += reference[:3]
            ret[:, 10:13] += reference[:3]
            ret[:, 13:16] += reference[:3]
            ret[:, 16:19] += reference[:3]
        elif feature_version == "v3":
            ret[:, 7:10] += reference[:3]
            ret[:, 10:13] += reference[:3]
            ret[:, 13:16] += reference[:3]
        else:
            raise ValueError
    return ret


def normalize_features_torch(features, reference, feature_version):
    # features: [bs, num_obj, num_features]
    # reference: [bs, num_obj, 3]
    features[:, :, :3] -= reference
    if feature_version == "v1":
        features[:, :, 3:6] -= reference
        features[:, :, 6:9] -= reference
    elif feature_version == "v2":
        features[:, :, 7:10] -= reference
        features[:, :, 10:13] -= reference
        features[:, :, 13:16] -= reference
        features[:, :, 16:19] -= reference
    elif feature_version == "v3":
        features[:, :, 7:10] -= reference
        features[:, :, 10:13] -= reference
        features[:, :, 13:16] -= reference
    else:
        raise ValueError
    return features


def compute_scene_centroid(features_args, features_others, feature_version):
    if feature_version == "v1":
        if features_others.size > 0:
            all_aabbs = np.vstack((features_args[:, 3:9], features_others[:, 3:9]))
        else:
            all_aabbs = features_args[:, 3:9]
        scene_extrema = np.array(
            [np.amin(all_aabbs[:, :3], axis=0), np.amax(all_aabbs[:, 3:], axis=0)]
        )

    elif feature_version == "v2":
        if features_others.size > 0:
            all_oabbs = np.vstack((features_args[:, 7:19], features_others[:, 7:19]))
            all_oabbs = all_oabbs.reshape((-1, 4, 3))
        else:
            all_oabbs = features_args[:, 7:19].reshape((-1, 4, 3))
        scene_extrema = np.array(
            [np.amin(all_oabbs, axis=(0, 1)), np.amax(all_oabbs, axis=(0, 1))]
        )
    elif feature_version == "v3":
        if features_others.size > 0:
            all_oabbs = np.vstack((features_args[:, 7:16], features_others[:, 7:16]))
            all_oabbs = all_oabbs.reshape((-1, 3, 3))
        else:
            all_oabbs = features_args[:, 7:16].reshape((-1, 3, 3))
        scene_extrema = np.array(
            [np.amin(all_oabbs, axis=(0, 1)), np.amax(all_oabbs, axis=(0, 1))]
        )
    else:
        raise ValueError
    centroid = np.mean(scene_extrema, axis=0)
    return centroid


def compute_scene_centroid_torch(features, feature_version):
    # features: [batch_size, num_objects, num_features]
    if feature_version == "v1":
        all_aabbs = features[:, :, 3:9]
        min_corner = torch.min(all_aabbs[:, :, :3], dim=1, keepdim=True)[0]
        max_corner = torch.max(all_aabbs[:, :, 3:], dim=1, keepdim=True)[0]
        scene_extrema = torch.cat((min_corner, max_corner), dim=1)

    elif feature_version == "v2":
        if features_others.size > 0:
            all_oabbs = np.vstack((features_args[:, 7:19], features_others[:, 7:19]))
            all_oabbs = all_oabbs.reshape((-1, 4, 3))
        else:
            all_oabbs = features_args[:, 7:19].reshape((-1, 4, 3))
        scene_extrema = np.array(
            [np.amin(all_oabbs, axis=(0, 1)), np.amax(all_oabbs, axis=(0, 1))]
        )
    elif feature_version == "v3":
        if features_others.size > 0:
            all_oabbs = np.vstack((features_args[:, 7:16], features_others[:, 7:16]))
            all_oabbs = all_oabbs.reshape((-1, 3, 3))
        else:
            all_oabbs = features_args[:, 7:16].reshape((-1, 3, 3))
        scene_extrema = np.array(
            [np.amin(all_oabbs, axis=(0, 1)), np.amax(all_oabbs, axis=(0, 1))]
        )
    else:
        raise ValueError
    centroid = torch.mean(scene_extrema, dim=1, keepdim=True)
    return centroid


def scale_features(features, scale, feature_version):
    ret = features
    if len(ret) > 0:
        if feature_version == "v1":
            if ret.ndim == 2:
                ret[:, :3] *= scale
                ret[:, 3:9] *= scale
                ret[:, 13:19] *= scale
            elif ret.ndim == 3:
                ret[:, :, 3] *= scale
                ret[:, :, 3:9] *= scale
                ret[:, :, 13:19] *= scale
            else:
                raise ValueError
        else:
            raise NotImplementedError
    return ret


def build_graph(node_features_args, node_features_others, label, demo_id):
    num_args = node_features_args.shape[0]
    num_others = (
        node_features_others.shape[0] if node_features_others is not None else 0
    )

    edge_list = list()

    # Create edges between argument nodes
    for ii in range(0, num_args - 1):
        for jj in range(ii + 1, num_args):
            edge_list.append([ii, jj])
            edge_list.append([jj, ii])

    # Create edges between argument and other nodes
    for ii in range(0, num_args):
        for jj in range(num_args, num_args + num_others):
            edge_list.append([ii, jj])
            edge_list.append([jj, ii])

    # Create tensors
    edge_index = torch.tensor(edge_list, dtype=torch.long)
    edge_index = edge_index.t().contiguous()
    if node_features_others is not None:
        x = torch.cat((node_features_args, node_features_others), dim=0)
    else:
        x = node_features_args
    assert x.shape[0] == num_args + num_others

    # Create data point
    data = GraphData(x=x, edge_index=edge_index, y=label, demo_id=demo_id)
    return data


def compute_aabb(oabb, pos, orient):
    # Get all vertices of oabb
    mg = np.array(np.meshgrid([0, 1], [0, 1], [0, 1])).T.reshape(-1, 3)
    vertices = np.array([[oabb[conf[i], i] for i in range(3)] for conf in mg])

    # Bring into correct orientation
    rot = R.from_quat(orient)
    rot_vertices = rot.apply(vertices)

    # Shift to correct position
    final_vertices = rot_vertices + pos

    # Compute aabb
    aabb = np.array([np.amin(final_vertices, axis=0), np.amax(final_vertices, axis=0)])

    return aabb


def compute_aabb_graph(oabb, pos, orient):
    oabb_reshape = np.reshape(oabb, (-1, 2, 3))

    # Get all vertices of oabb
    mg = np.array(np.meshgrid([0, 1], [0, 1], [0, 1])).T.reshape(-1, 3)
    vertices = np.array(
        [
            [[oabb_reshape[j, conf[i], i] for i in range(3)] for conf in mg]
            for j in range(oabb_reshape.shape[0])
        ]
    )

    # Bring into correct orientation
    rot = R.from_quat(orient)
    rot_vertices = np.array([rot[i].apply(x) for i, x in enumerate(vertices)])

    # Shift to correct position
    final_vertices = np.array(
        [rot_vertices[i, :, :] + pos[i, :] for i in range(rot_vertices.shape[0])]
    )

    # Compute aabb
    aabb = np.array(
        [
            [np.amin(final_vertices[i], axis=0), np.amax(final_vertices[i], axis=0)]
            for i in range(rot_vertices.shape[0])
        ]
    )
    aabb = aabb.reshape((oabb.shape[0], -1))

    return aabb


def compute_aabb_torch(oabb, pos, orient, device):
    shifted_oabb_vertices = shift_oabb(oabb, pos, orient, device)

    # Compute aabb
    lower_bounds, _ = torch.min(shifted_oabb_vertices, dim=2)
    upper_bounds, _ = torch.max(shifted_oabb_vertices, dim=2)
    aabb = torch.cat((lower_bounds, upper_bounds), dim=2)
    return aabb


@lru_cache
def get_combination_indices():
    idx_x, idx_y, idx_z = (0, 3), (1, 4), (2, 5)
    all_combinations_wrong_order = np.array(
        list(itertools.product(idx_z, idx_x, idx_y))
    )
    all_combinations = np.array(
        (
            all_combinations_wrong_order[:, 1],
            all_combinations_wrong_order[:, 2],
            all_combinations_wrong_order[:, 0],
        )
    ).transpose()
    all_combinations_list = [
        tuple(all_combinations[i, :]) for i in range(all_combinations.shape[0])
    ]
    return all_combinations_list


def get_vertices_from_oabb(oabb, dtype):
    all_combinations = get_combination_indices()
    vertices = torch.zeros((oabb.size(0), oabb.size(1), 8, 3), dtype=dtype)
    for i, combination in enumerate(all_combinations):
        vertices[:, :, i, :] = oabb[:, :, combination]
    return vertices


def shift_oabb(oabb, pos, orient, device):
    """
    Args:
        oabb: [batch_size, num_objects, 6]
        pos: [batch_size, num_objects, 3]
        orient: [batch_size, num_objects, 4], scalar last
        device: str
    """

    # Get all vertices of oabb
    vertices = get_vertices_from_oabb(oabb, dtype=pos.dtype)

    # Old implementation that is 22x slower
    # mg = np.array(np.meshgrid([0, 1], [0, 1], [0, 1])).T.reshape(-1, 3)
    # vertices = torch.tensor(
    #     [
    #         [
    #             [[oabb[j, k, m + conf[m] * 3] for m in range(3)] for conf in mg]
    #             for k in range(oabb.shape[1])
    #         ]
    #         for j in range(oabb.shape[0])
    #     ]
    # )

    vertices = torch.cat(
        (
            vertices,
            torch.zeros(vertices.shape[0], vertices.shape[1], vertices.shape[2], 1),
        ),
        dim=3,
    ).to(device)

    # Bring into correct orientation
    conj = torch.ones_like(orient)
    conj[:, :, :3] = torch.mul(conj[:, :, :3], -1)
    orient_conj = torch.mul(orient, conj)
    orient_mat = orient.reshape(
        (vertices.shape[0], vertices.shape[1], 1, -1)
    ).expand_as(vertices)
    conj_orient_mat = orient_conj.reshape(
        (vertices.shape[0], vertices.shape[1], 1, -1)
    ).expand_as(vertices)

    # https://math.stackexchange.com/questions/40164/how-do-you-rotate-a-vector-by-a-unit-quaternion/535223
    tmp = hamilton_product(orient_mat, vertices)
    rotated_vertices = hamilton_product(tmp, conj_orient_mat)
    rotated_vertices = rotated_vertices[:, :, :, :3]

    # Shift to correct position
    broadcast_pos = torch.unsqueeze(pos, 2).expand_as(rotated_vertices)
    final_vertices = torch.add(rotated_vertices, broadcast_pos)

    return final_vertices


def hamilton_product(a, b):
    # https://de.wikipedia.org/wiki/Quaternion#Quaternionenmultiplikation_als_Skalar-_und_Kreuzprodukt
    vec = (
        torch.mul(torch.unsqueeze(a[:, :, :, 3], dim=3), b[:, :, :, :3])
        + torch.mul(torch.unsqueeze(b[:, :, :, 3], dim=3), a[:, :, :, :3])
        + torch.cross(a[:, :, :, :3], b[:, :, :, :3], dim=3)
    )
    scalar = a[:, :, :, 3] * b[:, :, :, 3] - torch.sum(
        torch.mul(a[:, :, :, :3], b[:, :, :, :3]), dim=3
    )
    scalar = torch.unsqueeze(scalar, dim=3)
    res = torch.cat((vec, scalar), dim=3)
    return res


def apply_rotation_to_vertices(vertices, orient, inv=False):
    orient_scalar_first = (
        orient[:, :, [3, 0, 1, 2]]
        .unsqueeze(2)
        .expand(vertices.size(0), vertices.size(1), 8, 4)
    )
    if inv:
        orient_scalar_first = transforms.quaternion_invert(orient_scalar_first)
    vertices_rotated = transforms.quaternion_apply(orient_scalar_first, vertices)
    return vertices_rotated


def shift_oabb_pt3d(oabb, pos, orient):
    """
    Args:
        oabb: [batch_size, num_objects, 6]
        pos: [batch_size, num_objects, 3]
        orient: [batch_size, num_objects, 4], scalar last
    """

    vertices = get_vertices_from_oabb(oabb, dtype=pos.dtype)
    vertices_rotated = apply_rotation_to_vertices(vertices, orient)
    vertices_shifted = vertices_rotated + pos.unsqueeze(2)
    return vertices_shifted
