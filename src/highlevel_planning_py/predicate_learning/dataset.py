import os
import pickle

import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data.in_memory_dataset import InMemoryDataset as GraphDataset

from highlevel_planning_py.predicate_learning import data_utils as du
import highlevel_planning_py.predicate_learning.training_utils as tu


def build_sklearn_dataset(dataset):
    num_samples = len(dataset)
    num_features = dataset.get_num_features()
    num_args = dataset.get_num_arguments()
    X = np.zeros((num_samples, num_args * num_features))
    y = -1 * np.ones(num_samples)
    for i in range(num_samples):
        sample = dataset[i]
        X[i, :] = sample.x.numpy().flatten()
        y[i] = sample.y.numpy().flatten()
    return X, y


class PredicateDatasetBase:
    def __init__(
        self,
        data_dir,
        data_session_id,
        predicate_name,
        use_tensors,
        normalization_method,
        positive_only=False,
        label_arg_objects=False,
        feature_version: str = "v1",
        data_scaling: float = None,
    ):
        self.predicate_name = predicate_name
        self.is_graph_data = False
        self.use_tensors = use_tensors
        self.normalize = normalization_method != "none"
        self.normalization_method = normalization_method
        self.label_arg_objects = label_arg_objects
        self.feature_version = feature_version

        self.data_scaling = data_scaling
        # This was introduced to pass the scaling factor to downstream tasks.

        feature_file = os.path.join(
            data_dir,
            predicate_name,
            "features",
            data_session_id,
            f"{predicate_name}_{feature_version}.pkl",
        )
        with open(feature_file, "rb") as f:
            self.features, self.relations, self.meta_data = pickle.load(f)
        if positive_only:
            self.demo_ids = [
                rel for rel, props in self.relations.items() if props["label"]
            ]
        else:
            self.demo_ids = list(self.relations.keys())

    def get_single_by_demo_id(self, demo_id, use_tensors=None):
        if use_tensors is None:
            use_tensors = self.use_tensors
        label = int(self.relations[demo_id]["label"])

        features_args = np.array(
            [
                self.features.iloc[i, :].values
                for i in self.relations[demo_id]["arguments"]
            ]
        )
        if self.label_arg_objects:
            features_args = np.concatenate(
                (features_args, np.ones((features_args.shape[0], 1))), axis=1
            )

        features_others = np.array(
            [self.features.iloc[i, :].values for i in self.relations[demo_id]["others"]]
        )
        if self.label_arg_objects and features_others.ndim > 1:
            features_others = np.concatenate(
                (features_others, np.zeros((features_others.shape[0], 1))), axis=1
            )

        if self.data_scaling is not None:
            features_args = du.scale_features(
                features_args, self.data_scaling, self.feature_version
            )
            features_others = du.scale_features(
                features_others, self.data_scaling, self.feature_version
            )

        if self.normalize:
            if self.normalization_method == "first_arg":
                ref = np.copy(features_args[0, :])
            elif self.normalization_method == "scene_center":
                ref = du.compute_scene_centroid(
                    features_args, features_others, self.feature_version
                )
            else:
                raise ValueError("Invalid normalization method selected")
            features_args = du.normalize_features(
                features_args, ref, self.feature_version
            )
            features_others = du.normalize_features(
                features_others, ref, self.feature_version
            )
        if use_tensors:
            label = torch.tensor(label).float()
            features_args = torch.from_numpy(features_args).float()
            features_others = torch.from_numpy(features_others).float()
        obj_names = {
            "args": self.relations[demo_id]["argument_names"],
            "others": self.relations[demo_id]["other_names"],
        }
        return features_args, features_others, label, demo_id, obj_names

    def get_single(self, idx, use_tensors=None):
        demo_id = self.demo_ids[idx]
        return self.get_single_by_demo_id(demo_id, use_tensors)

    def get_num_features(self):
        return self.meta_data["num_features"]

    def get_num_arguments(self):
        return self.meta_data["num_arguments"]

    def get_balance_info(self):
        labels = [
            ent["label"]
            for demo_id, ent in self.relations.items()
            if demo_id in self.demo_ids
        ]
        unique, counts = np.unique(labels, return_counts=True)
        print(f"Balance info: {dict(zip(unique, counts))}")


class PredicateDataset(PredicateDatasetBase, Dataset):
    def __init__(self, config: tu.DatasetConfig):
        super(PredicateDataset, self).__init__(
            config.data_dir,
            config.data_session_id,
            config.predicate_name,
            config.use_tensors,
            config.normalization_method,
            config.positive_only,
            config.label_arg_objects,
            config.feature_version,
            config.data_scaling,
        )

    def __len__(self):
        return len(self.demo_ids)

    def __getitem__(self, idx):
        features_args, _, label, demo_id, _ = self.get_single(idx)
        return features_args, label, demo_id


class PredicateGraphDataset(GraphDataset):
    def __init__(self, config: tu.DatasetConfig):
        self.is_graph_data = True
        self.config = config

        pred_dir = os.path.join(config.data_dir, config.predicate_name)
        if not os.path.isfile(
            os.path.join(
                pred_dir,
                "features",
                config.data_session_id,
                f"{config.predicate_name}_{config.feature_version}.pkl",
            )
        ):
            raise ValueError("Features for requested predicate don't exist.")

        dir_name = config.predicate_name
        dir_name += "-pos" if config.positive_only else "-posneg"
        dir_name += f"-{config.feature_version}"
        dir_name += f"-{config.normalization_method}"
        dir_name += "-surr" if config.include_surrounding else "-nosurr"
        dataset_dir = os.path.join(
            pred_dir, "datasets", config.data_session_id, dir_name
        )
        os.makedirs(dataset_dir, exist_ok=True)

        self.base_ds = PredicateDatasetBase(
            config.data_dir,
            config.data_session_id,
            config.predicate_name,
            use_tensors=True,
            normalization_method=config.normalization_method,
            positive_only=config.positive_only,
            label_arg_objects=config.label_arg_objects,
            feature_version=config.feature_version,
        )

        super().__init__(root=dataset_dir)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ["data.pt"]

    def get_num_features(self):
        return self.base_ds.get_num_features()

    def get_num_arguments(self):
        return self.base_ds.get_num_arguments()

    def download(self):
        pass

    def process(self):
        self.base_ds.get_balance_info()

        # Read data into huge `Data` list.
        data_list = list()
        for i in range(len(self.base_ds.demo_ids)):
            data = self.get_single_graph(
                i, include_surrounding=self.config.include_surrounding
            )
            data_list.append(data)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    def get_single(self, idx, use_tensors=None):
        return self.base_ds.get_single(idx, use_tensors)

    def get_single_by_demo_id(self, demo_id, use_tensors=None):
        return self.base_ds.get_single_by_demo_id(demo_id, use_tensors)

    def get_single_graph(self, idx, include_surrounding=True):
        if type(idx) is int:
            node_features_args, node_features_others, label, demo_id, _ = self.base_ds.get_single(
                idx, use_tensors=True
            )
        elif type(idx) is str:
            node_features_args, node_features_others, label, demo_id, _ = self.base_ds.get_single_by_demo_id(
                idx, use_tensors=True
            )
        else:
            raise ValueError
        if not include_surrounding:
            node_features_others = None
        data = du.build_graph(node_features_args, node_features_others, label, demo_id)
        return data


def main():
    data_dir = os.path.join(os.path.expanduser("~"), "Data", "highlevel_planning")
    pred_dir = os.path.join(data_dir, "predicates", "data")
    data_session_id = "220608-190256_demonstrations_features"

    config = tu.DatasetConfig(
        "cpu",
        pred_dir,
        data_session_id,
        "on_clutter",
        splits={"train": 0.9, "val": 0.1},
        target_model="gen",
        normalization_method="scene_center",
        feature_version="v1",
    )

    # ds = PredicateDataset(config)
    # ds.get_balance_info()
    # ds.get_single(5)

    ds = PredicateGraphDataset(config)
    ds.get_single(7)
    print("bla")


if __name__ == "__main__":
    main()
