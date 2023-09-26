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

import argparse
import os
import tracemalloc
from typing import Dict, List, Any
from datetime import datetime
import time
import sys
import pickle
import pprint
from copy import deepcopy
import numpy as np
import atexit
from collections import defaultdict
from dataclasses import dataclass, field

from tensorboardX import SummaryWriter
import torch
from torch import nn
import logging
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
import torch_geometric as pyg
from torch_geometric.graphgym.utils.io import dict_to_json, dict_to_tb

from highlevel_planning_py.predicate_learning import models_pc_encoding
from highlevel_planning_py.tools_pl import util


@dataclass
class ConfigBase:
    device: str  # One of cpu, cuda

    def verify(self):
        pass

    def to_dict(self):
        ret = dict()
        for item in dir(self):
            val = getattr(self, item)
            if item.startswith("__") or callable(val):
                continue
            if isinstance(val, ConfigBase):
                ret[item] = val.to_dict()
            else:
                ret[item] = val
        return ret


@dataclass
class TrainingConfig(ConfigBase):
    random_seed: int = 1

    model_type: str = "mlp"  # One of mlp, gnn, hybrid
    model_version: str = "v1"

    predicate_name: str = "on_supporting_ig"
    dataset_id: str = "220325_demonstrations_features_03"
    dataset_size: int = -1
    dataset_feature_version: str = "v1"
    data_label_arg_objects: bool = False
    set_requires_grad: bool = False
    loss_aggregation: str = "sum"

    batch_size: int = 5
    training_style: str = "iterations"  # One of iterations or epochs

    # if training_style is "epochs":
    num_epochs: int = -1
    # if training_style is "iterations":
    num_gen_iterations: int = -1
    # if training style should be ctat
    training_schedule: dict = field(default_factory=dict)

    disc_iters_per_gen_iter: int = 1  # How many disc iterations for each gen iteration

    dim_noise: int = 1
    optimizer: str = "adam"  # One of adam, rmsprop
    learning_rate: float = 0.001
    use_lr_scheduler: bool = True

    pass_label_to_disc: bool = False
    pass_label_to_gen: bool = False

    eval_period: int = 1
    train_log_period: int = 20
    save_model_period: int = 0  # 0 means don't save model, any int > 0 determines save interval
    save_img_period: int = 0  # 0 means don't save image, any int > 0 determines save interval
    save_img_fixed_scene_id: str = ""
    use_tracemalloc: bool = False

    loss: str = "minimax"  # One of minimax, wasserstein, mixed

    # Clipper is used if loss="wasserstein"
    clipper_type: str = "clamp"  # One of clamp, norm, none
    clipper_radius: float = 2.0

    # If Wasserstein loss is used, gradient penalty can be used instead of clipping
    gradient_penalty: bool = False
    gradient_penalty_lambda: float = 10.0

    # Loss components for generator
    gen_loss_components: str = "disc"  # One of disc, class, both

    def verify(self):
        if self.loss == "wasserstein":
            if self.gradient_penalty:
                assert self.clipper_type == "none"
        else:
            assert self.clipper_type == "none" and not self.gradient_penalty

        # if self.loss == "mixed":
        #     assert self.disc_predict_label

        if self.training_style == "iterations":
            assert self.num_gen_iterations > 0
        elif self.training_style == "epochs":
            assert self.num_epochs > 0
        elif self.training_style == "schedule":
            assert len(self.training_schedule) > 0
        else:
            raise ValueError("Invalid training style")

        for step in self.training_schedule:
            assert step == "ct" or step == "at"


@dataclass
class GNNNetworkConfig(ConfigBase):
    dim_in: int
    dim_out: int
    layers: List[int]  # [layers_pre_mp, layers_mp, layers_post_mp]
    dim_inner: List[int]  # [dim_inner_pre_mp, dim_inner_mp, dim_inner_post_mp]

    mp_layer_type: str = "gatconv"
    graph_pooling: str = "none"
    post_processing: bool = True
    initialize_with: str = ""
    custom: dict = field(default_factory=dict)
    eps: float = 1e-8

    def verify(self):
        assert len(self.layers) in [0, 3]


@dataclass
class MLPNetworkConfig(ConfigBase):
    dim_out: int
    layers: List[int]
    custom: dict = field(default_factory=dict)
    dim_in: int = 0
    initialize_with: str = ""
    feature_version: str = "v1"
    activation_type: str = "relu"
    eps: float = 1e-8

    def verify(self):
        pass


@dataclass
class HybridNetworkConfig(ConfigBase):
    num_argument_objects: int
    num_features: int
    config_encoder: Any
    config_main_net: MLPNetworkConfig

    encoder_type: str = "own"  # One of "own", "gat"
    feature_version: str = "v1"

    initialize_with: str = ""

    inflate_features: tuple = field(default_factory=tuple)
    inflate_to: int = 10

    custom: dict = field(default_factory=dict)

    def complete(self):
        if len(self.inflate_features) == 2:
            num_features_inflated = (
                self.num_features
                - (self.inflate_features[1] - self.inflate_features[0])
                + self.inflate_to
                if self.inflate_features[1] > self.inflate_features[0]
                else self.num_features
            )
        else:
            num_features_inflated = self.num_features

        # Set config_main_net.dim_in
        self.config_main_net.dim_in = self.num_argument_objects * num_features_inflated
        if len(self.config_encoder.layers) > 0:
            if self.encoder_type in ["gat", "own"]:
                if self.config_encoder.layers[2] > 0:
                    self.config_main_net.dim_in += self.config_encoder.dim_out
                else:
                    assert self.config_encoder.layers[1] > 0
                    self.config_main_net.dim_in += self.config_encoder.dim_inner[1]
            elif self.encoder_type == "mlp":
                self.config_main_net.dim_in += self.config_encoder.dim_out
            else:
                raise ValueError

        # Set config_encoder.dim_in
        if self.encoder_type == "mlp":
            self.config_encoder.dim_in = (
                self.num_argument_objects + 1
            ) * num_features_inflated

    def verify(self):
        self.config_encoder.verify()


@dataclass
class HybridTransformerNetworkConfig(ConfigBase):
    # General
    num_argument_objects: int
    num_features: int

    # MLP
    config_mlp: MLPNetworkConfig

    feature_version: str = "v1"

    initialize_with: str = ""

    # Encoder
    num_heads: int = 8
    dim_feedforward: int = 64
    num_encoder_layers: int = 3

    custom: dict = field(default_factory=dict)

    def complete(self):
        pass


@dataclass
class DatasetConfig(ConfigBase):
    data_dir: str
    data_session_id: str
    predicate_name: str
    splits: Dict[str, float]
    target_model: str
    normalization_method: str = "first_arg"  # One of first_arg, scene_center, none
    positive_only: bool = False
    include_surrounding: bool = True
    label_arg_objects: bool = False
    use_tensors: bool = True
    feature_version: str = "v1"
    dataset_size: int = -1
    encoder_id: str = ""
    data_scaling: float = None
    custom: dict = field(default_factory=dict)


def compute_split_indices(split_proportions, dataset_size):
    if len(split_proportions) == 1:
        splits = (list(range(dataset_size)),)
    elif "test" in split_proportions:
        splits = train_test_split(
            dataset_size,
            val_split=split_proportions["val"],
            test_split=split_proportions["test"],
            shuffle_dataset=True,
        )
    elif "val" in split_proportions:
        splits = train_test_split(
            dataset_size, val_split=split_proportions["val"], shuffle_dataset=True
        )
    else:
        raise RuntimeError
    return splits


def get_graph_data_loaders(
    dataset, split_proportions: dict, batch_size, dataset_size=-1
):
    loaders = dict()
    split_names = list(split_proportions.keys())
    if dataset_size > len(dataset):
        print(
            f"WARNING: selected dataset size larger than dataset length."
            f"Using dataset length."
        )
    ds = (
        len(dataset)
        if dataset_size == -1 or dataset_size > len(dataset)
        else dataset_size
    )
    splits = compute_split_indices(split_proportions, ds)
    for i in range(len(split_names)):
        if len(splits[i]) > 0:
            loaders[split_names[i]] = pyg.loader.DataLoader(
                dataset[splits[i]], batch_size=batch_size, shuffle=True
            )
    dim_in = dataset.data.x.shape[1] if hasattr(dataset.data.x, "shape") else 0
    return loaders, dim_in


def get_standard_data_loaders(
    dataset, split_proportions: dict, batch_size, dataset_size=-1
):
    loaders = dict()
    split_names = list(split_proportions.keys())
    if dataset_size > len(dataset):
        print(
            f"WARNING: selected dataset size larger than dataset length."
            f"Using dataset length."
        )
    ds = (
        len(dataset)
        if dataset_size == -1 or dataset_size > len(dataset)
        else dataset_size
    )
    splits = compute_split_indices(split_proportions, ds)
    for i in range(len(split_names)):
        subset = torch.utils.data.Subset(dataset, splits[i])
        loaders[split_names[i]] = torch.utils.data.DataLoader(
            subset, batch_size=batch_size, shuffle=True
        )
    dim_in = dataset.features.shape[1]
    return loaders, dim_in


def train_test_split(
    dataset_size, val_split, test_split=0.0, shuffle_dataset=False, random_seed=12
):
    indices = list(range(dataset_size))
    test_split = int(np.floor(test_split * dataset_size))
    test_indices = indices[:test_split]
    indices = indices[test_split:]
    val_split = int(np.floor(val_split * dataset_size))
    if shuffle_dataset:
        if random_seed is not None:
            np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[val_split:], indices[:val_split]
    return train_indices, val_indices, test_indices


def set_printing(out_dir, screen_output=True):
    os.makedirs(out_dir, exist_ok=True)
    logging.root.handlers = list()
    logging_cfg = {"level": logging.INFO, "format": "%(message)s"}
    h_file = logging.FileHandler(os.path.join(out_dir, "logging.log"))
    if screen_output:
        h_stdout = logging.StreamHandler(sys.stdout)
        logging_cfg["handlers"] = [h_file, h_stdout]
    else:
        logging_cfg["handlers"] = [h_file]
    logging.basicConfig(**logging_cfg)


class TrainLogger:
    def __init__(self, name, out_dir, loss_aggregation, round_digits=4):
        self.name = name
        self.out_dir = os.path.join(out_dir, name)
        assert loss_aggregation in ["sum", "mean"]
        self.loss_aggregation = loss_aggregation
        self.round_digits = round_digits

        os.makedirs(self.out_dir, exist_ok=True)
        self.tb_writer = SummaryWriter(self.out_dir)
        self.data = None
        self._reset()

    def record_iteration(
        self, loss: dict, lr, batch_size, true=None, pred=None, ground_truth=None
    ):
        self.data["iter"] += 1
        if true is not None or pred is not None:
            assert true.shape[0] == pred.shape[0]
            self.data["true"].append(true)
            self.data["pred"].append(pred)
        self.data["size_current"] += batch_size
        for label in loss:
            if self.loss_aggregation == "sum":
                self.data["loss"][label] += loss[label]
            else:
                self.data["loss"][label] += loss[label] * batch_size
        self.data["lr"] = lr
        if ground_truth is not None:
            self.data["ground_truth"].append(ground_truth)

    def write_epoch(self, cur_epoch):
        if self.data["size_current"] == 0:
            return
        basic_stats = self._get_basic_stats()
        task_stats = self._get_classification_stats()
        special_stats = self._get_special_stats()
        epoch_stats = {"epoch": cur_epoch}
        stats = {**epoch_stats, **basic_stats, **task_stats, **special_stats}

        # Print
        logging.info(f"{self.name}: {stats}")
        # JSON
        dict_to_json(stats, os.path.join(self.out_dir, f"stats.json"))
        # Tensorboard
        dict_to_tb(stats, self.tb_writer, cur_epoch)

        self._reset()
        if "auc" in task_stats and "accuracy" in task_stats:
            return task_stats["auc"], task_stats["accuracy"]
        else:
            return None, None

    def close(self):
        self.tb_writer.close()

    def _reset(self):
        self.data = {
            "iter": 0,
            "size_current": 0,
            "loss": defaultdict(float),
            "lr": 0,
            "true": list(),
            "pred": list(),
            "ground_truth": list(),
        }

    def _round(self, inp):
        return np.around(inp, self.round_digits)

    def _get_basic_stats(self):
        stats = {
            "lr": self._round(self.data["lr"]),
            "size_current": self.data["size_current"],
        }
        loss_stats = {
            f"loss_{loss_label}": self._round(
                self.data["loss"][loss_label] / self.data["size_current"]
            )
            for loss_label in self.data["loss"]
        }
        return {**loss_stats, **stats}

    @staticmethod
    def _get_pred_int(pred_score):
        return (pred_score > 0.5).long()

    def _get_classification_stats(self):
        if len(self.data["true"]) > 0:
            true, pred_score = (
                torch.cat(self.data["true"]),
                torch.cat(self.data["pred"]),
            )
            pred_int = self._get_pred_int(pred_score)
            try:
                r_a_score = roc_auc_score(true, pred_score)
            except ValueError:
                r_a_score = 0.0
            stats = {
                "accuracy": self._round(accuracy_score(true, pred_int)),
                "precision": self._round(
                    precision_score(true, pred_int, zero_division=0)
                ),
                "recall": self._round(recall_score(true, pred_int)),
                "f1": self._round(f1_score(true, pred_int)),
                "auc": self._round(r_a_score),
            }
        else:
            stats = dict()
        return stats

    def _get_special_stats(self):
        if len(self.data["ground_truth"]) > 0:
            ground_truth = torch.cat(self.data["ground_truth"])
            ground_truth_score = torch.sum(ground_truth).item()
            ground_truth_score /= self.data["size_current"]
            stats = {"gt_performance": ground_truth_score}
        else:
            stats = dict()
        return stats


def is_nth_epoch(cur_epoch, n, max_epoch):
    """Determines if the model should be evaluated at the current epoch."""
    if n == -1:
        return False
    return cur_epoch % n == 0 or cur_epoch == 1 or cur_epoch == max_epoch


class RessourceLogger:
    def __init__(self, name, out_dir):
        self.name = name
        self.out_dir = os.path.join(out_dir, name)
        os.makedirs(self.out_dir, exist_ok=True)
        self.tb_writer = SummaryWriter(self.out_dir)

    def log_entry(self, epoch, value_dict):
        entry = {"epoch": epoch, **value_dict}
        dict_to_tb(entry, self.tb_writer, epoch)
        logging.info(f"{self.name}: {entry}")

    def close(self):
        self.tb_writer.close()


class TrainingSequenceBase:
    def __init__(self, model, learning_rate, device, out_dir, eval_period=1):
        self.model = model
        self.device = device
        self.eval_period = eval_period
        self.out_dir = out_dir

        self.loss_fcn = nn.BCEWithLogitsLoss(reduction="mean")
        params = self.model.parameters()
        params = filter(lambda p: p.requires_grad, params)
        self.optimizer = torch.optim.Adam(params, lr=learning_rate)
        self.loggers = {
            "train": TrainLogger("train", out_dir),
            "val": TrainLogger("val", out_dir),
            "test": TrainLogger("test", out_dir),
        }
        atexit.register(self._close_loggers)

    def train(self, loaders, num_epochs: int):
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=num_epochs
        )
        logging.info("Starting training")
        best_auc, best_acc, best_epoch = 0.0, 0.0, -1
        best_model_state = None
        last_auc, last_acc, epoch = 0.0, 0.0, -1
        last_model_state = None
        for epoch in range(num_epochs):
            self.train_epoch(loaders[0], scheduler)
            self.loggers["train"].write_epoch(epoch)
            if is_nth_epoch(epoch, self.eval_period, num_epochs):
                self.eval_epoch(loaders[1], "val")
                auc, acc = self.loggers["val"].write_epoch(epoch)
                if auc >= best_auc and acc > best_acc:
                    best_auc = auc
                    best_acc = acc
                    best_epoch = epoch
                    best_model_state = deepcopy(self.model.state_dict())
                if epoch == num_epochs - 1:
                    last_auc = auc
                    last_acc = acc
                    last_model_state = deepcopy(self.model.state_dict())
                self.eval_epoch(loaders[2], "test")
                self.loggers["test"].write_epoch(epoch)
        torch.save(
            best_model_state,
            os.path.join(
                self.out_dir,
                f"best_model_auc{best_auc:.2f}_acc{best_acc:.2f}_e{best_epoch}.pt",
            ),
        )
        torch.save(
            last_model_state,
            os.path.join(
                self.out_dir,
                f"last_model_auc{last_auc:.2f}_acc{last_acc:.2f}_e{epoch}.pt",
            ),
        )
        logging.info(
            f"Best AUC: {best_auc}, best acc: {best_acc}, best epoch: {best_epoch}"
        )
        logging.info("Training done")

    def train_epoch(self, dataloader, scheduler):
        raise NotImplementedError

    def eval_epoch(self, dataloader, split: str):
        raise NotImplementedError

    def _compute_loss(self, pred, true):
        pred = pred.squeeze(-1) if pred.ndim > 1 else pred
        true = true.squeeze(-1) if true.ndim > 1 else true
        true = true.float()
        loss = self.loss_fcn(pred, true)
        pred_score = torch.sigmoid(pred)
        return loss, pred_score

    def _close_loggers(self):
        for logger in self.loggers.values():
            logger.close()
        logging.info("Loggers closed")


def save_parameters(
    parameters: dict, filename, out_dir, parameters_obj=None, txt_only=False
):
    if not txt_only:
        filename_pkl = os.path.join(out_dir, f"{filename}.pkl")
        with open(filename_pkl, "wb") as f:
            if parameters_obj is None:
                pickle.dump(parameters, f)
            else:
                pickle.dump(parameters_obj, f)
    filename_txt = os.path.join(out_dir, f"{filename}.txt")
    with open(filename_txt, "w") as f:
        pprint.pprint(parameters, f, sort_dicts=False)


def load_parameters(out_dir, filename):
    filename_pkl = os.path.join(out_dir, f"{filename}.pkl")
    with open(filename_pkl, "rb") as f:
        data = pickle.load(f)
    return data


def setup_out_dir(
    predicate_dir,
    resume_training,
    resume_timestring,
    screen_output=True,
    training_type: str = "",
):
    if not resume_training:
        if len(resume_timestring) == 0:
            time_now = datetime.now()
            time_string = time_now.strftime("%y%m%d_%H%M%S")
            if len(training_type) > 0:
                time_string += f"_{training_type}"
        else:
            time_string = resume_timestring
        out_dir = os.path.join(predicate_dir, "training", f"{time_string}")
        assert not os.path.isdir(out_dir), "Not resuming, but out_dir exists."
    else:
        out_dir = os.path.join(predicate_dir, "training", f"{resume_timestring}")
        assert os.path.isdir(out_dir)
    set_printing(out_dir, screen_output)
    return out_dir


def create_model(mdl_type, mdl_params, resume, device, out_dir, predicate_dir, label):
    # Load hyperparameters
    if resume:
        mdl_params = load_parameters(out_dir, f"parameters_{label}")
    else:
        save_parameters(
            mdl_params.to_dict(), f"parameters_{label}", out_dir, mdl_params
        )
    mdl_params.verify()

    # Create model
    model = mdl_type(mdl_params)
    logging.info(f"{label} model:")
    logging.info(model)
    num_parameters = pyg.graphgym.utils.comp_budget.params_count(model)
    logging.info(f"Number of model parameters: {num_parameters}")

    # Load weights
    latest_iteration = 0
    if resume:
        latest_state_file, latest_iteration = util.get_latest_weights(out_dir, label)
        latest_state_file = os.path.join(out_dir, "models", latest_state_file)
        model.load_state_dict(torch.load(latest_state_file))
    elif len(mdl_params.initialize_with) > 0:
        state_file = os.path.join(
            predicate_dir,
            "training",
            mdl_params.initialize_with,
            "models",
            f"{label}_model_final.pt",
        )
        assert os.path.isfile(state_file), f"{state_file} not found."
        model.load_state_dict(torch.load(state_file))
        logging.info(f"Initialized {label} model from run {mdl_params.initialize_with}")
    model.to(device)

    return model, num_parameters, latest_iteration


def get_empty_stats():
    stats = {
        "time_elapsed": 0.0,
        "time_manual_classifier": 0.0,
        "total_iterations": 0,
        "training_schedule": list(),
        "num_sessions": 0,
        "max_mem_gb": 0.0,
    }
    return stats


def training_admin_beginning(
    run_config, out_dir, resume_training, resume_additional_it
):
    if not resume_training:
        run_config.verify()
        stats = get_empty_stats()
        if run_config.training_style == "schedule":
            stats["training_schedule"] = list()
        save_parameters(
            run_config.to_dict(), "parameters_training", out_dir, run_config
        )
    else:
        run_config = load_parameters(out_dir, "parameters_training")
        if run_config.training_style == "schedule":
            run_config.training_schedule = resume_additional_it
        elif run_config.training_style == "iterations":
            run_config.num_gen_iterations = resume_additional_it
        elif run_config.training_style == "epochs":
            run_config.num_epochs = resume_additional_it
        try:
            stats = load_parameters(out_dir, "stats")
        except FileNotFoundError:
            stats = get_empty_stats()
            with open(os.path.join(out_dir, "stats_incomplete.txt"), "w") as f:
                f.write(
                    "Stats incomplete. There is at least one previous run that was not logged."
                )
    return stats, run_config


def run_training(training_sequence, run_config: TrainingConfig, stats, out_dir):
    # Training
    if run_config.use_tracemalloc:
        tracemalloc.start()
    start_time = time.time()
    training_sequence.train(stats["total_iterations"], start_time)
    end_time = time.time()
    if run_config.use_tracemalloc:
        logging.info(f"Memory: {tracemalloc.get_traced_memory()}")
        tracemalloc.stop()

    # Save stats
    stats["time_elapsed"] += end_time - start_time
    if training_sequence.ground_truth_classifier is not None:
        stats[
            "time_manual_classifier"
        ] += training_sequence.ground_truth_classifier.total_time
    if run_config.training_style == "iterations":
        stats["total_iterations"] += run_config.num_gen_iterations
    elif run_config.training_style == "epochs":
        stats["total_iterations"] += run_config.num_epochs
    elif run_config.training_style == "schedule":
        for step in run_config.training_schedule.items():
            stats["total_iterations"] += step[1]
            stats["training_schedule"].append(step)
    stats["num_sessions"] += 1
    stats["max_mem_gb"] = max(stats["max_mem_gb"], training_sequence.mem_ps_gb_max)
    save_parameters(stats, "stats", out_dir)
    logging.info("Saved stats")


def get_dataloader(
    resume_training, out_dir, dataset_type, batch_size, dataset_config_in
):
    config_id_str = f"{dataset_config_in.target_model}"
    for split in dataset_config_in.splits:
        config_id_str += f"_{split}"
    if resume_training:
        dataset_config = load_parameters(out_dir, f"parameters_data_{config_id_str}")
        dataset_config.data_dir = dataset_config_in.data_dir
    else:
        dataset_config = dataset_config_in
        save_parameters(
            dataset_config.to_dict(),
            f"parameters_data_{config_id_str}",
            out_dir,
            dataset_config,
        )

    dataset = dataset_type(dataset_config)
    if dataset.is_graph_data:
        loaders, dim_in = get_graph_data_loaders(
            dataset,
            dataset_config.splits,
            batch_size,
            dataset_size=dataset_config.dataset_size,
        )
    else:
        loaders, dim_in = get_standard_data_loaders(
            dataset,
            dataset_config.splits,
            batch_size,
            dataset_size=dataset_config.dataset_size,
        )
    return loaders, dim_in, dataset


def set_assert(data, label, value):
    if label in data:
        assert data[label] == value
    else:
        data[label] = value
    return data


def setup_data(
    loaders,
    datasets,
    dimensions,
    resume_training,
    out_dir,
    batch_size: int,
    dataset_configs: List[DatasetConfig],
    dataset_types,
):
    for i, config in enumerate(dataset_configs):
        if config.target_model in loaders:
            for split in config.splits:
                assert (
                    split not in loaders[config.target_model]
                ), "Conflicting dataset specs"
        this_loaders, this_dimension, this_dataset = get_dataloader(
            resume_training, out_dir, dataset_types[i], batch_size, config
        )
        for split, loader in this_loaders.items():
            if len(loader.dataset) < batch_size:
                raise RuntimeError(f"Batch size larger than dataset size for {split}.")
            loaders[config.target_model][split] = loader
        dimensions = set_assert(dimensions, config.target_model, this_dimension)
        for split in config.splits:
            datasets[config.target_model][split] = this_dataset


class ModelSplitter:
    def __init__(self, model: nn.Module, output_idx: int):
        self.mdl = model
        self.output_idx = output_idx


def get_activation_function(activation_type: str):
    if activation_type == "relu":
        return nn.ReLU
    elif activation_type == "prelu":
        return nn.PReLU
    elif activation_type == "tanh":
        return nn.Tanh
    elif activation_type == "leakyrelu":
        return nn.LeakyReLU
    elif activation_type == "elu":
        return nn.ELU
    else:
        raise ValueError


def parse_general_args(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--paths", type=str, choices=["local", "home", "euler"], default="local"
    )
    parser.add_argument(
        "--id_string",
        action="store",
        help="Give this run a name. If left empty, a current time stamp will be used.",
        default="",
        type=str,
    )
    parser.add_argument("--gpu", type=util.string_to_bool, default=False)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--explicit_exit_handler", action="store_true")
    parser.add_argument(
        "--screen_output",
        action="store_true",
        help="if given, loggers will print to screen",
    )
    parser.add_argument("--predicate_name", type=str, default="on_supporting_ig")
    parser.add_argument(
        "--dataset_id", type=str, default="220831-175353_demonstrations_features"
    )
    parser.add_argument("--random_seed", action="store", type=int, default=12)
    parser.add_argument("--batch_size", action="store", type=int, default=16)
    parser.add_argument("--learning_rate", action="store", type=float, default=0.001)
    parser.add_argument("--dataset_size", action="store", type=int, default=-1)
    parser.add_argument(
        "--run_gt_classifier", action="store", type=util.string_to_bool, default=True
    )
    parser.add_argument("--evaluate_gen", action="store_true")
    parser.add_argument("--evaluate_class", action="store_true")


def parse_general_ctat_args(parser: argparse.ArgumentParser):
    parser.add_argument("--num_class_it", action="store", type=int, default=0)
    parser.add_argument("--num_adversarial_it", action="store", type=int, default=0)
    parser.add_argument(
        "--feature_version", action="store", choices=["v1", "v2", "v3"], default="v1"
    )
    parser.add_argument(
        "--gen_loss_components",
        action="store",
        choices=["both", "disc", "class"],
        default="both",
    )
    parser.add_argument(
        "--data_normalization_class",
        action="store",
        choices=["scene_center", "first_arg", "none"],
        default="scene_center",
    )
    parser.add_argument(
        "--data_normalization_disc_gen",
        action="store",
        choices=["scene_center", "first_arg", "none"],
        default="first_arg",
    )
    parser.add_argument(
        "--gen_normalize_output", type=util.string_to_bool, default=False
    )
    parser.add_argument("--gen_pass_label", type=util.string_to_bool, default=False)
    parser.add_argument("--disc_pass_label", type=util.string_to_bool, default=False)
    parser.add_argument(
        "--loss_aggregation", type=str, default="sum", choices=["sum", "mean"]
    )
    parser.add_argument("--init_class_with", action="store", type=str, default="")
    parser.add_argument(
        "--save_period", type=str, choices=["normal", "long"], default="normal"
    )


def parse_arguments_hybrid(parser: argparse.ArgumentParser, model_type: str):
    parser.set_defaults(model_type=model_type)
    parser.add_argument("--model_version", type=str, default="v2", choices=["v1", "v2"])
    parser.add_argument(
        "--include_surrounding", type=util.string_to_bool, default=False
    )
    parser.add_argument("--scene_encoding_dim", type=int, default=20)
    parser.add_argument(
        "--class_encoder_type", type=str, choices=["own", "gat", "mlp"], default="mlp"
    )
    parser.add_argument(
        "--class_encoder_layers", type=util.string_to_list, default=[1, 1, 1]
    )
    parser.add_argument(
        "--class_gnn_dim_inner", type=util.string_to_list, default=[10, 10, 10]
    )
    parser.add_argument(
        "--class_main_net_layers", type=util.string_to_list, default=[12, 12]
    )
    parser.add_argument(
        "--disc_encoder_type", type=str, choices=["own", "gat", "mlp"], default="mlp"
    )
    parser.add_argument(
        "--disc_encoder_layers", type=util.string_to_list, default=[1, 1, 1]
    )
    parser.add_argument(
        "--disc_gnn_dim_inner", type=util.string_to_list, default=[10, 10, 10]
    )
    parser.add_argument(
        "--disc_main_net_layers", type=util.string_to_list, default=[12, 12]
    )
    parser.add_argument(
        "--gen_encoder_type", type=str, choices=["own", "gat", "mlp"], default="mlp"
    )
    parser.add_argument(
        "--gen_encoder_layers", type=util.string_to_list, default=[1, 1, 1]
    )
    parser.add_argument(
        "--gen_gnn_dim_inner", type=util.string_to_list, default=[10, 10, 10]
    )
    parser.add_argument(
        "--gen_main_net_layers", type=util.string_to_list, default=[12, 12]
    )


def parse_arguments_mlp(parser: argparse.ArgumentParser):
    parser.set_defaults(model_type="mlp")
    parser.add_argument("--model_version", type=str, default="v1", choices=["v1"])
    parser.add_argument("--class_layers", type=util.string_to_list, default=[12, 12])
    parser.add_argument("--disc_layers", type=util.string_to_list, default=[12, 12])
    parser.add_argument("--gen_layers", type=util.string_to_list, default=[12, 12])
    parser.add_argument(
        "--class_activation",
        type=str,
        default="relu",
        choices=["relu", "prelu", "tanh", "leakyrelu", "elu"],
    )
    parser.add_argument(
        "--disc_activation",
        type=str,
        default="relu",
        choices=["relu", "prelu", "tanh", "leakyrelu", "elu"],
    )
    parser.add_argument(
        "--gen_activation",
        type=str,
        default="relu",
        choices=["relu", "prelu", "tanh", "leakyrelu", "elu"],
    )


def load_encoder_decoder(run_dir, device):
    run_config = load_parameters(run_dir, "arguments_training")
    meta_parameters = load_parameters(
        os.path.join(run_dir, "models"), "meta_parameters"
    )

    all_models = os.listdir(os.path.join(run_dir, "models"))
    all_models.sort()
    encoder_weights = sorted([model for model in all_models if "encoder" in model])[-1]
    decoder_weights = sorted([model for model in all_models if "decoder" in model])[-1]
    encoder, decoder = None, None
    if run_config.encoder_type == "pointnet2":
        if run_config.decoder_type in ["pointnet2"]:
            encoder = models_pc_encoding.Pointnet2Encoder(
                run_config.encoding_dim, run_config.model_size_enc, autoencoder=True
            )
        else:
            encoder = models_pc_encoding.Pointnet2Encoder(
                run_config.encoding_dim, run_config.model_size_enc, autoencoder=False
            )
    elif run_config.encoder_type == "conv":
        encoder = models_pc_encoding.ConvEncoder(
            run_config.encoding_dim, model_config=run_config.model_size_enc
        )
    if run_config.decoder_type == "pointnet2":
        decoder = models_pc_encoding.Pointnet2Decoder(
            run_config.encoding_dim, run_config.model_size_dec
        )
    elif run_config.decoder_type == "conv":
        decoder = models_pc_encoding.ConvDecoder(
            run_config.encoding_dim,
            meta_parameters["num_points"],
            model_config=run_config.model_size_dec,
        )
    elif run_config.decoder_type == "fc":
        decoder = models_pc_encoding.FullyConnectedDecoder(
            run_config.encoding_dim,
            meta_parameters["num_points"],
            model_config=run_config.model_size_dec,
        )
    encoder.load_state_dict(
        torch.load(os.path.join(run_dir, "models", encoder_weights))
    )
    decoder.load_state_dict(
        torch.load(os.path.join(run_dir, "models", decoder_weights))
    )
    encoder.to(device)
    decoder.to(device)
    return encoder, decoder


def exit_handler(start_time, stats, out_dir, loggers):
    for label, logger in loggers.items():
        logger.close()
    logging.info("Loggers closed")
