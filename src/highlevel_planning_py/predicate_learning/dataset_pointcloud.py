import os
import os.path as osp
import gzip
import pickle
from typing import Union, List, Tuple
from plyfile import PlyData
import numpy as np
import torch
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.transforms import BaseTransform, Center, RandomScale

from highlevel_planning_py.predicate_learning import training_utils as tu
from highlevel_planning_py.predicate_learning import data_utils as du
from highlevel_planning_py.predicate_learning.dataset import PredicateDatasetBase


class NormalizeScaleSingle(BaseTransform):
    r"""Centers and normalizes node positions to the interval :math:`(-scale_target, scale_target)`
    (functional name: :obj:`normalize_scale`).
    """

    def __init__(self, scale_target: float = 1.0, return_size: bool = False):
        self.center = Center()
        self.scale_target = scale_target
        self.return_size = return_size

    def __call__(self, data):
        if type(data) is list:
            assert len(data) == 1
            data = data[0]
        data = self.center(data)

        max_size = data.pos.abs().max()
        scale = (self.scale_target / max_size) * 0.999999
        data.pos = data.pos * scale

        if self.return_size:
            return data, max_size
        else:
            return data


class NormalizeScaleAllData(BaseTransform):
    def __init__(
        self,
        aggregation_method: str,
        given_scale: int = None,
        scale_target: float = 1.0,
    ):
        self.center = Center()
        self.aggregation_method = aggregation_method
        self.given_scale = given_scale
        self.scale_target = scale_target
        self.all_data = True
        self.return_size = False

    def __call__(self, data_list: List):
        data_list = [self.center(data) for data in data_list]

        if self.given_scale is None:
            all_sizes = list()
            for data_single in data_list:
                all_sizes.append(data_single.pos.abs().max().item())
            if "median" in self.aggregation_method:
                scale = self.scale_target / np.median(all_sizes)
            elif "mean" in self.aggregation_method:
                scale = self.scale_target / np.mean(all_sizes)
            elif "max" in self.aggregation_method:
                scale = self.scale_target / np.max(all_sizes)
            else:
                raise ValueError
        else:
            scale = self.given_scale

        for data_single in data_list:
            data_single.pos = data_single.pos * scale

        return data_list, scale

    def __repr__(self):
        return f"{self.__class__.__name__}_{self.aggregation_method}()"


class SinglePointcloudDataset(InMemoryDataset):
    def __init__(self, config: tu.DatasetConfig):
        self.is_graph_data = True
        self.config = config
        pred_dir = osp.join(config.data_dir, config.predicate_name)
        self._raw_dir_path = str(
            osp.join(pred_dir, "pointclouds_individual", config.data_session_id)
        )
        self.num_points = config.custom["num_points"]

        given_scale = None
        scale_target = (
            config.custom["scale_target"] if "scale_target" in config.custom else 1.0
        )
        if self.config.normalization_method == "individual_max":
            pre_transform = NormalizeScaleSingle(scale_target)
        elif "all_data" in self.config.normalization_method:
            given_scale = (
                config.custom["given_scale"] if "given_scale" in config.custom else None
            )
            pre_transform = NormalizeScaleAllData(
                self.config.normalization_method,
                given_scale=given_scale,
                scale_target=scale_target,
            )
        elif self.config.normalization_method == "none":
            pre_transform = None
        else:
            raise ValueError

        dir_name = (
            f"{config.predicate_name}_pointclouds-single_norm_"
            f"{self.config.normalization_method}_p{self.num_points}_"
            f"scl{scale_target}"
        )
        dataset_dir = osp.join(pred_dir, "datasets", config.data_session_id, dir_name)
        os.makedirs(dataset_dir, exist_ok=True)

        if (
            "augment_scale" in config.custom
            and config.custom["augment_scale"] is not None
        ):
            aug_scale = RandomScale(config.custom["augment_scale"])
        else:
            aug_scale = None

        super().__init__(dataset_dir, pre_transform=pre_transform, transform=aug_scale)
        self.data, self.slices, self.scale = torch.load(self.processed_paths[0])

        if given_scale is not None:
            assert (
                self.scale == given_scale
            ), f"Stored dataset with different scaling. To continue, delete {dataset_dir}"

    @property
    def raw_dir(self) -> str:
        return self._raw_dir_path

    @property
    def raw_file_names(self):
        return [f"point_clouds-{self.num_points}.gz"]

    @property
    def processed_file_names(self):
        return [f"point_clouds-{self.num_points}.pt"]

    def download(self):
        pass

    def process(self):
        raw_file = self.raw_paths[0]
        with gzip.open(raw_file, "rb") as f:
            all_data = pickle.load(f)

        # Read data into huge `Data` list.
        if self.config.use_tensors:
            data_list = [
                Data(pos=torch.from_numpy(points).float(), num_nodes=points.shape[0])
                for points in all_data
            ]
        else:
            data_list = [
                Data(pos=points, num_nodes=points.shape[0]) for points in all_data
            ]

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        scale = None
        if self.pre_transform is not None:
            if hasattr(self.pre_transform, "all_data"):
                data_list, scale = self.pre_transform(data_list)
            else:
                data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices, scale), self.processed_paths[0])


def load_ply(filename):
    ply_data = PlyData.read(filename)
    points = ply_data["vertex"]
    points = np.vstack([points["x"], points["y"], points["z"]]).T
    return points.astype("float32")


class SinglePointcloudDatasetShapenet(InMemoryDataset):
    def __init__(self, config: tu.DatasetConfig):
        self.is_graph_data = True
        self.config = config

        assert config.normalization_method == "none"

        if config.custom["train"]:
            dataset_dir = osp.join(config.data_dir, "shapenet_pointclouds", "train_val")
            self._raw_dir_path = osp.join(config.custom["shapenet_dir"], "train_val")
        else:
            dataset_dir = osp.join(config.data_dir, "shapenet_pointclouds", "test")
            self._raw_dir_path = osp.join(config.custom["shapenet_dir"], "test")
        os.makedirs(dataset_dir, exist_ok=True)

        pc_paths = list()
        for path, subdir, files in os.walk(self._raw_dir_path):
            for name in files:
                p = osp.join(path, name)
                assert p.endswith(".ply")
                pc_paths.append(p)
        self.pc_paths = pc_paths

        self.scale = 1.0

        # Determine number of points
        x = load_ply(pc_paths[0])
        self.num_points = x.shape[0]
        del x

        if config.custom["augment_scale"] is not None:
            aug_scale = RandomScale(config.custom["augment_scale"])
        else:
            aug_scale = None

        super().__init__(dataset_dir, pre_transform=None, transform=aug_scale)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self) -> str:
        return self._raw_dir_path

    @property
    def raw_file_names(self):
        return self.pc_paths

    @property
    def processed_file_names(self):
        return ["point_clouds.pt"]

    def download(self):
        pass

    def process(self):
        data_list = list()
        for path in self.pc_paths:
            x = load_ply(path)
            dp = Data(pos=torch.from_numpy(x).float(), num_nodes=x.shape[0])
            data_list.append(dp)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        # scale = None
        # if self.pre_transform is not None:
        #     if hasattr(self.pre_transform, "all_data"):
        #         data_list, scale = self.pre_transform(data_list)
        #     else:
        #         data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


class PredicatePointcloudDataset(InMemoryDataset):
    """
    This is a dataset that contains all point clouds in every scene.
    It should be used when aiming to encode the clouds online, training the system end to end.
    """

    def __init__(self, config: tu.DatasetConfig, given_scale: float = None):
        self.is_graph_data = True
        self.config = config
        self.given_scale = given_scale

        if not self.config.use_tensors:
            raise NotImplementedError("Only implemented for tensors as data structure")

        pred_dir = osp.join(config.data_dir, config.predicate_name)
        dir_name = f"{config.predicate_name}_pointclouds"
        dir_name += "-pos_only" if config.positive_only else ""
        dir_name += "-surr" if config.include_surrounding else "-nosurr"
        dataset_dir = osp.join(pred_dir, "datasets", config.data_session_id, dir_name)
        os.makedirs(dataset_dir, exist_ok=True)

        self.raw_dir_path = osp.join(pred_dir, "pointclouds", config.data_session_id)
        self._raw_data = os.listdir(self.raw_dir_path)
        self._raw_data.sort()

        self.demo_ids = [fname.split(".")[0] for fname in self._raw_data]

        super().__init__(root=dataset_dir)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self) -> str:
        return self.raw_dir_path

    @property
    def raw_file_names(self):
        return self._raw_data

    @property
    def processed_file_names(self):
        return ["data.pt"]

    def download(self):
        pass

    def process(self):
        point_normalizer = NormalizeScaleAllData("max", given_scale=self.given_scale)

        data_list = list()
        num_args = None
        for demo_idx, raw_path in enumerate(self.raw_paths):
            with gzip.open(raw_path, "rb") as f:
                load_data = pickle.load(f)

            # Make sure number of arguments is consistent
            if num_args is None:
                num_args = len(load_data["arg_clouds"])
            else:
                assert num_args == len(load_data["arg_clouds"])

            pc_data_list = [
                Data(pos=torch.from_numpy(points).float(), num_nodes=points.shape[0])
                for points in load_data["arg_clouds"]
            ]
            pc_data_list.extend(
                [
                    Data(
                        pos=torch.from_numpy(points).float(), num_nodes=points.shape[0]
                    )
                    for points in load_data["other_clouds"]
                ]
            )
            positions = torch.stack(
                [points.pos.mean(dim=-2) for points in pc_data_list]
            )

            # Normalize scale
            pc_data_list, _ = point_normalizer(pc_data_list)

            # Normalize positions
            if self.config.normalization_method == "first_arg":
                ref = positions[0, :].expand_as(positions)
            elif self.config.normalization_method == "scene_center":
                ref = torch.mean(positions, dim=0, keepdim=True).expand_as(positions)
            else:
                raise ValueError("Invalid normalization method selected")
            positions -= ref

            point_tensor = torch.stack(
                [points.pos.reshape(-1) for points in pc_data_list]
            )
            combined_tensor = torch.cat((positions, point_tensor), dim=1)
            node_features_args = combined_tensor[:num_args, :]
            node_features_others = (
                combined_tensor[num_args:, :]
                if combined_tensor.shape[0] > num_args
                else None
            )
            data = du.build_graph(
                node_features_args,
                node_features_others,
                load_data["label"],
                self.demo_ids[demo_idx],
            )

            if self.pre_filter is not None and not self.pre_filter(data):
                continue

            if self.pre_transform is not None:
                data = self.pre_transform(data)

            data_list.append(data)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(osp.join(self.processed_dir, f"data_{idx}.pt"))
        return data


class PredicateEncodingGraphDataset(InMemoryDataset):
    def __init__(self, config: tu.DatasetConfig):
        self.is_graph_data = True
        self.config = config

        feature_file = os.path.join(
            config.data_dir,
            config.predicate_name,
            "pointclouds_encoded",
            config.data_session_id,
            config.encoder_id,
            "pointcloud_features.pkl",
        )
        if not os.path.isfile(feature_file):
            raise ValueError("Pointcloud encodings don't exist.")

        with open(feature_file, "rb") as f:
            self.features, self.relations, self.meta_data = pickle.load(f)
        if config.positive_only:
            self.demo_ids = [
                rel for rel, props in self.relations.items() if props["label"]
            ]
        else:
            self.demo_ids = list(self.relations.keys())

        pred_dir = os.path.join(config.data_dir, config.predicate_name)
        dir_name = "pc_encodings"
        dir_name += "-pos" if config.positive_only else "-posneg"
        dir_name += f"-{config.normalization_method}"
        dir_name += "-surr" if config.include_surrounding else "-nosurr"
        dataset_dir = os.path.join(
            pred_dir, "datasets", config.data_session_id, config.encoder_id, dir_name
        )
        os.makedirs(dataset_dir, exist_ok=True)

        # Add base dataset to make manual features available for debugging and visualizations
        self.base_ds = PredicateDatasetBase(
            config.data_dir,
            config.data_session_id,
            config.predicate_name,
            use_tensors=True,
            normalization_method=config.normalization_method,
            positive_only=config.positive_only,
            label_arg_objects=config.label_arg_objects,
            feature_version=config.feature_version,
            data_scaling=self.meta_data["data_scaling"],
        )

        super().__init__(root=dataset_dir)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> Union[str, List[str], Tuple]:
        return []

    @property
    def processed_file_names(self):
        return ["data.pt"]

    def get_num_features(self):
        return self.meta_data["num_features"]

    def get_num_arguments(self):
        return self.meta_data["num_arguments"]

    def download(self):
        pass

    def process(self):
        data_list = list()
        for i in range(len(self.demo_ids)):
            data = self.get_single_graph(i, self.config.include_surrounding)
            data_list.append(data)

        if self.pre_filter is not None:
            data_list = [data_ for data_ in data_list if self.pre_filter(data_)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    def get_single(self, idx: int, use_tensors=None):
        return self.base_ds.get_single(idx, use_tensors)

    def get_single_graph(self, idx: int, include_surrounding=True):
        if type(idx) is int:
            demo_id = self.demo_ids[idx]
        elif type(idx) is str:
            demo_id = idx
        else:
            raise ValueError
        arg_idx = self.relations[demo_id]["arguments"]
        features_args = self.features[arg_idx, :]
        if not include_surrounding:
            features_others = None
        else:
            other_idx = self.relations[demo_id]["others"]
            features_others = self.features[other_idx, :]

        # Normalize
        if self.config.normalization_method != "none":
            if self.config.normalization_method == "first_arg":
                ref = torch.clone(features_args[0, :3])
            elif self.config.normalization_method == "scene_center":
                ref = torch.mean(
                    torch.cat((features_args[:, :3], features_others[:, :3]), dim=0),
                    dim=0,
                )
            else:
                raise ValueError("Invalid normalization method")
            features_args[:, :3] -= ref.expand(features_args.shape[0], -1)
            if features_others is not None:
                features_others[:, :3] -= ref.expand(features_others.shape[0], -1)
        label = torch.tensor(int(self.relations[demo_id]["label"])).float()
        data = du.build_graph(features_args, features_others, label, demo_id)
        return data


if __name__ == "__main__":
    data_dir = os.path.join(os.path.expanduser("~"), "Data", "highlevel_planning")
    predicate_dir = os.path.join(data_dir, "predicates", "data")
    data_session_id = "220831-175353_demonstrations_features"

    # configur = tu.DatasetConfig(
    #     "cpu",
    #     predicate_dir,
    #     data_session_id,
    #     "on_clutter",
    #     splits={"train": 0.9, "val": 0.1},
    #     target_model="enc",
    #     normalization_method="all_data_max",
    #     use_tensors=True,
    #     # dataset_size=2000,
    #     custom={"num_points": 2048},
    # )
    # ds = SinglePointcloudDataset(configur)
    # # ds = PredicatePointcloudDataset(configur, given_scale=0.5)
    # print(len(ds))
    # # print(f"Scale: {ds.scale}")

    # configur = tu.DatasetConfig(
    #     "cpu",
    #     predicate_dir,
    #     data_session_id,
    #     "on_clutter",
    #     splits={"train": 0.9, "test": 0.1},
    #     target_model="enc",
    #     normalization_method="scene_center",
    #     encoder_id="220914_180008_12_--predicate_name-on_clutter_--encoder_type-conv_--decoder_type-fc_--dataset_normalization-all_data_max",
    # )
    # ds = PredicateEncodingGraphDataset(configur)

    # configur2 = tu.DatasetConfig(
    #     "cpu",
    #     predicate_dir,
    #     data_session_id,
    #     "on_clutter",
    #     splits={"train": 0.9, "test": 0.1},
    #     target_model="enc",
    #     normalization_method="scene_center",
    #     encoder_id="230105_151151_pcenc",
    # )
    # ds2 = PredicateEncodingGraphDataset(configur2)

    configur = tu.DatasetConfig(
        "cpu",
        predicate_dir,
        data_session_id,
        "on_clutter",
        splits={"train": 0.9, "test": 0.1},
        target_model="enc",
        custom={
            "shapenet_dir": osp.join(
                osp.expanduser("~"),
                "Data",
                "latent_3d_points",
                "data",
                "shape_net_core_uniform_samples_2048_split",
            ),
            "train": True,
            "augment_scale": (0.1, 2.0),
        },
    )
    ds = SinglePointcloudDatasetShapenet(configur)

    print("bla")
