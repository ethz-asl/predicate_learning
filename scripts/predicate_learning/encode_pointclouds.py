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
import gzip
import pickle
import torch
from torch_geometric.data import Data
from tqdm import tqdm

from highlevel_planning_py.predicate_learning import training_utils as tu
from highlevel_planning_py.predicate_learning.dataset_pointcloud import (
    NormalizeScaleSingle,
    NormalizeScaleAllData,
    SinglePointcloudDataset,
)
from highlevel_planning_py.predicate_learning.dataset import PredicateDataset
from highlevel_planning_py.tools_pl.path import get_path_dict

PATHS_SELECTOR = "local"


def encode_clouds(cloud_list, normalizer, encoder):
    encodings = list()
    positions = list()
    centroids = list()
    orientations = list()
    sizes = list()
    for cloud in cloud_list:
        pc_data = Data(pos=torch.from_numpy(cloud).float(), num_nodes=cloud.shape[0])
        position = pc_data.pos.mean(dim=-2).reshape(1, -1)
        positions.append(position)
        centroid = (pc_data.pos.max(dim=-2)[0] + pc_data.pos.min(dim=-2)[0]) / 2.0
        centroid = centroid.reshape(1, -1)
        centroids.append(centroid)
        if type(normalizer) is NormalizeScaleAllData:
            pc_data = normalizer([pc_data])[0][0]
        elif type(normalizer) is NormalizeScaleSingle:
            pc_data, this_size = normalizer(pc_data)
            if normalizer.return_size:
                sizes.append(this_size)
        else:
            raise NotImplementedError
        encoding = encoder.forward(pc_data)
        encodings.append(encoding.detach().cpu())
        orientations.append(torch.tensor([0, 0, 0, 1]).reshape(1, -1))
    assert len(positions) == len(encodings)
    if len(encodings) == 0:
        return None, None, None, None, None
    else:
        sizes = None if len(sizes) == 0 else torch.stack(sizes).view(-1, 1)
        return (
            torch.cat(encodings),
            torch.cat(positions),
            torch.cat(orientations),
            torch.cat(centroids),
            sizes,
        )


def main(
    predicate_name,
    data_session_id,
    encoder_run_id,
    recompute_scaling: bool = False,
    scale_separate: bool = False,
    scale_target: float = 1.0,
):
    """
    Args:
        predicate_name:
        data_session_id:
        encoder_run_id:
        recompute_scaling:  if set to false, will take scaling from encoder meta-data. if true,
                            will rescale the data.
        scale_separate:     scale every object individually to a box, and save the original size
                            with the other features.
        scale_target:       the target size to scale to, when recompute_scaling or scale_separate is set to true.

    Returns:

    """

    paths = get_path_dict(PATHS_SELECTOR)

    predicate_dir = os.path.join(paths["data_dir"], "predicates")
    dataset_dir = os.path.join(predicate_dir, "data")

    # Prepare where to save encoded data
    out_dir = os.path.join(
        dataset_dir,
        predicate_name,
        "pointclouds_encoded",
        data_session_id,
        encoder_run_id,
    )
    os.makedirs(out_dir, exist_ok=True)
    out_filename = "pointcloud_features.pkl"
    out_path = os.path.join(out_dir, out_filename)
    if os.path.isfile(out_path):
        print(
            f"Feature file already exists. Aborting. Delete and run again for {encoder_run_id}"
        )
        return

    # Load encoder
    run_dir = os.path.join(predicate_dir, "training", encoder_run_id)
    encoder, _ = tu.load_encoder_decoder(run_dir, "cpu")
    encoder_meta_params = tu.load_parameters(
        os.path.join(run_dir, "models"), "meta_parameters"
    )
    num_points = encoder_meta_params["num_points"]

    if not scale_separate:
        if recompute_scaling:
            ds_config = tu.DatasetConfig(
                "cpu",
                dataset_dir,
                data_session_id,
                predicate_name,
                splits={"train": 1.0},
                target_model="",
                normalization_method="all_data_max",
                use_tensors=True,
                custom={"num_points": num_points, "scale_target": scale_target},
            )
            ds = SinglePointcloudDataset(ds_config)
            data_scaling_factor = ds.scale
            del ds
        else:
            data_scaling_factor = encoder_meta_params["data_scaling"]

        # Prepare data normalization
        normalizer = NormalizeScaleAllData("", given_scale=data_scaling_factor)
    else:
        data_scaling_factor = 1.0
        normalizer = NormalizeScaleSingle(scale_target, return_size=True)

    # Define data to be encoded
    pointcloud_dir = os.path.join(
        dataset_dir,
        predicate_name,
        "pointclouds",
        data_session_id,
        f"num_points-{num_points}",
    )
    filenames_pc = os.listdir(pointcloud_dir)
    filenames_pc.sort()

    # Data structures
    features = torch.tensor([])
    relations = dict()
    meta_data = {
        "data_scaling": data_scaling_factor,
        "recompute_scaling": recompute_scaling,
        "scale_separate": scale_separate,
        "scale_target": scale_target,
    }

    # For debug purposes also load feature dataset
    ds_config = tu.DatasetConfig(
        "cpu",
        dataset_dir,
        data_session_id,
        predicate_name,
        splits={"train": 1.0},
        target_model="gen",
        normalization_method="none",
    )
    ds_features = PredicateDataset(ds_config)

    # Encode point clouds
    for filename_raw_pc in tqdm(filenames_pc):
        demo_id = filename_raw_pc.split(".")[0]

        path_raw_pc = os.path.join(pointcloud_dir, filename_raw_pc)
        with gzip.open(path_raw_pc, "rb") as f:
            load_data = pickle.load(f)

        relations[demo_id] = {"label": load_data["label"]}

        arg_encodings, arg_positions, arg_orientations, arg_centroids, arg_sizes = encode_clouds(
            load_data["arg_clouds"], normalizer, encoder
        )

        f_args, f_others, _, _, _ = ds_features.get_single_by_demo_id(demo_id)

        # Scale positions to stay consistent
        arg_centroids = data_scaling_factor * arg_centroids

        start_idx = features.shape[0]
        if not scale_separate:
            pos_and_features = torch.cat(
                (arg_centroids, arg_orientations, arg_encodings), dim=1
            )
        else:
            pos_and_features = torch.cat(
                (arg_centroids, arg_orientations, arg_sizes, arg_encodings), dim=1
            )
        features = torch.cat((features, pos_and_features))
        relations[demo_id]["arguments"] = tuple(
            range(start_idx, start_idx + pos_and_features.shape[0])
        )

        if "num_arguments" not in meta_data:
            meta_data["num_arguments"] = arg_encodings.shape[0]
        else:
            assert meta_data["num_arguments"] == arg_encodings.shape[0]

        other_encodings, other_positions, other_orientations, other_centroids, other_sizes = encode_clouds(
            load_data["other_clouds"], normalizer, encoder
        )
        if other_encodings is not None:
            other_centroids = data_scaling_factor * other_centroids
            start_idx = features.shape[0]
            if not scale_separate:
                pos_and_features = torch.cat(
                    (other_centroids, other_orientations, other_encodings), dim=1
                )
            else:
                pos_and_features = torch.cat(
                    (other_centroids, other_orientations, other_sizes, other_encodings),
                    dim=1,
                )
            features = torch.cat((features, pos_and_features))
            relations[demo_id]["others"] = tuple(
                range(start_idx, start_idx + pos_and_features.shape[0])
            )
        else:
            relations[demo_id]["others"] = tuple()

    if "num_features" not in meta_data:
        meta_data["num_features"] = features.shape[1]

    save_data = (features, relations, meta_data)
    with open(out_path, "wb") as f:
        pickle.dump(save_data, f)

    print("Finished successfully")


if __name__ == "__main__":
    predicate_name_ = "inside_drawer"
    data_session_id_ = "230606-124817_demonstrations_features"
    specs_multiple = [
        # {
        #     "encoder_run_id": "230320_103300_01_pcenc_own_prescaled_1_0",
        #     "recompute_scaling": False,
        #     "scale_separate": False,
        #     "scale_target": 1.0,
        # },
        # {
        #     "encoder_run_id": "230320_103300_02_pcenc_own_augment_1_0",
        #     "recompute_scaling": True,
        #     "scale_separate": False,
        #     "scale_target": 1.0,
        # },
        # {
        #     "encoder_run_id": "230320_103300_03_pcenc_shapenet_1_0",
        #     "recompute_scaling": True,
        #     "scale_separate": False,
        #     "scale_target": 1.0,
        # },
        # {
        #     "encoder_run_id": "230320_103300_04_pcenc_own_prescaled_0_5",
        #     "recompute_scaling": False,
        #     "scale_separate": False,
        #     "scale_target": 1.0,
        # },
        # {
        #     "encoder_run_id": "230320_103300_05_pcenc_own_augment_0_5",
        #     "recompute_scaling": True,
        #     "scale_separate": False,
        #     "scale_target": 0.5,
        # },
        # {
        #     "encoder_run_id": "230320_103300_06_pcenc_shapenet_0_5",
        #     "recompute_scaling": True,
        #     "scale_separate": False,
        #     "scale_target": 0.5,
        # },
        # {
        #     "encoder_run_id": "230320_103300_07_pcenc_finetuned_1_0",
        #     "recompute_scaling": True,
        #     "scale_separate": False,
        #     "scale_target": 1.0,
        # },
        # {
        #     "encoder_run_id": "230320_103300_08_pcenc_finetuned_0_5",
        #     "recompute_scaling": True,
        #     "scale_separate": False,
        #     "scale_target": 0.5,
        # },
        # {
        #     "encoder_run_id": "230320_103300_09_pcenc_own_indi_1_0",
        #     "recompute_scaling": False,
        #     "scale_separate": True,
        #     "scale_target": 1.0,
        # },
        # {
        #     "encoder_run_id": "230320_103300_10_pcenc_shapenet_indi_1_0",
        #     "recompute_scaling": False,
        #     "scale_separate": True,
        #     "scale_target": 1.0,
        # },
        # {
        #     "encoder_run_id": "230320_103300_11_pcenc_own_indi_0_5",
        #     "recompute_scaling": False,
        #     "scale_separate": True,
        #     "scale_target": 0.5,
        # },
        # {
        #     "encoder_run_id": "230320_103300_12_pcenc_shapenet_indi_0_5",
        #     "recompute_scaling": False,
        #     "scale_separate": True,
        #     "scale_target": 0.5,
        # },
        {
            "encoder_run_id": "230608_102122_01_pcenc_--predicate_name-inside_drawer_--dataset_normalization-all_data_max_--data_prescale_target-1.0",
            "recompute_scaling": False,
            "scale_separate": False,
            "scale_target": 1.0,
        },
        {
            "encoder_run_id": "230608_102122_02_pcenc_--predicate_name-inside_drawer_--dataset_normalization-all_data_max_--data_prescale_target-0.5",
            "recompute_scaling": False,
            "scale_separate": False,
            "scale_target": 1.0,
        },
        {
            "encoder_run_id": "230608_102122_03_pcenc_--predicate_name-inside_drawer_--dataset_normalization-individual_max_--data_prescale_target-1.0",
            "recompute_scaling": False,
            "scale_separate": True,
            "scale_target": 1.0,
        },
        {
            "encoder_run_id": "230608_102122_04_pcenc_--predicate_name-inside_drawer_--dataset_normalization-individual_max_--data_prescale_target-0.5",
            "recompute_scaling": False,
            "scale_separate": True,
            "scale_target": 0.5,
        },
        {
            "encoder_run_id": "230608_102122_05_pcenc_--predicate_name-inside_drawer_--augment_training_data-[0.05,2.0]",
            "recompute_scaling": True,
            "scale_separate": False,
            "scale_target": 1.0,
        },
        {
            "encoder_run_id": "230608_102122_06_pcenc_--predicate_name-inside_drawer_--augment_training_data-[0.05,1.0]",
            "recompute_scaling": True,
            "scale_separate": False,
            "scale_target": 0.5,
        },
    ]

    for spec in specs_multiple:
        main(predicate_name_ + "_test", data_session_id_, **spec)
        main(predicate_name_, data_session_id_, **spec)
