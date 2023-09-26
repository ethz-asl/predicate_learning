#!/usr/bin/env python

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
from typing import Dict
import time
import tracemalloc
import subprocess
import logging
import shutil
import numpy as np
from matplotlib import pyplot as plt
import torch_geometric as pyg

from highlevel_planning_py.predicate_learning.models_gnn import (
    WeightClipperNorm,
    WeightClipperClamp,
    propagate_over_batches,
)
from highlevel_planning_py.predicate_learning import training_utils as tu
from highlevel_planning_py.predicate_learning.predicate_learning_server import SimServer
from highlevel_planning_py.tools_pl import util
from highlevel_planning_py.tools_pl.logger_stdout import LoggerStdout

import torch
from torch import nn
from torch.utils.data import DataLoader as TorchDataLoader


def check_weights_equal(w_cl_bef, w_cl_after, label):
    weights_equal = True
    unequal_idx = list()
    for i in range(len(w_cl_after)):
        if not torch.equal(w_cl_bef[i], w_cl_after[i]):
            weights_equal = False
            unequal_idx.append(i)
    print(f"Weights unchanged {label} it: {weights_equal} {unequal_idx}")


class DatasetAdapter:
    def __init__(self, device):
        self.device = device

    def get_batch_size(self, batch):
        raise NotImplementedError

    def interpolate(self, real_sample, fake_sample):
        raise NotImplementedError

    def compute_gradient_norm(
        self, in_interpolated, out_interpolated, label, grad_outputs
    ):
        raise NotImplementedError

    def prepare_batch(self, batch):
        raise NotImplementedError

    @staticmethod
    def get_new_features_args(generated_scenes, dataset=None):
        raise NotImplementedError

    def init_fixed_input(self, num_fixed, fixed_features_args, fixed_scene_id, dataset):
        raise NotImplementedError

    def evaluate_gt_classifier(
        self, batch, clf, dataset, get_reason=False, demo_ids=None
    ):
        raise NotImplementedError

    def get_index_by_id(self, dataset, sample_id):
        raise NotImplementedError

    def get_manual_features_vis(self, batch, dataset):
        raise NotImplementedError


class DatasetAdapterGraph(DatasetAdapter):
    def __init__(self, device):
        super().__init__(device)

    def get_batch_size(self, batch):
        return batch.num_graphs

    def interpolate(self, real_sample, fake_sample):
        if real_sample.num_nodes < fake_sample.num_nodes:
            smaller_sample = real_sample
            larger_sample = fake_sample
        else:
            smaller_sample = fake_sample
            larger_sample = real_sample
        if smaller_sample.num_graphs > 1:
            interpolated = pyg.data.batch.Batch.from_data_list(
                smaller_sample.to_data_list()
            ).to(self.device)
        else:
            interpolated = pyg.data.batch.Batch.from_data_list([smaller_sample]).to(
                self.device
            )
        eta = torch.rand(smaller_sample.num_graphs).to(self.device)
        eta_x = propagate_over_batches(interpolated, eta, self.device)
        interpolated.x = torch.mul(eta_x, smaller_sample.x) + torch.mul(
            (1 - eta_x), larger_sample.x[: smaller_sample.num_nodes, :]
        )
        interpolated.x.requires_grad_(True)
        eta_y = eta.expand(smaller_sample.num_graphs)
        interpolated.y = torch.mul(eta_y, smaller_sample.y) + torch.mul(
            (1 - eta_y), larger_sample.y
        )
        interpolated.y.requires_grad_(True)
        return interpolated

    def compute_gradient_norm(
        self, in_interpolated, out_interpolated, label, grad_outputs
    ):
        net_input = (
            (in_interpolated.x, label) if label is not None else in_interpolated.x
        )

        gradients = torch.autograd.grad(
            out_interpolated,
            net_input,
            grad_outputs=grad_outputs,
            create_graph=True,
            retain_graph=True,
            allow_unused=True,
        )
        if in_interpolated.num_graphs > 1:
            s_dict = in_interpolated.ptr
            gradient_norm_x = [
                torch.linalg.norm(gradients[0][s_dict[i] : s_dict[i + 1], :])
                for i in range(len(s_dict) - 1)
            ]
        else:
            gradient_norm_x = [torch.linalg.norm(gradients[0])]
        gradient_norm_x = torch.stack(gradient_norm_x)
        if len(gradients) > 1 and gradients[1] is not None:
            gradient_norm_y = torch.linalg.norm(
                torch.unsqueeze(gradients[1], dim=1), dim=1
            )
            gradient_norm = gradient_norm_x + gradient_norm_y
        else:
            gradient_norm = gradient_norm_x
        return gradient_norm

    def prepare_batch(self, batch):
        batch.to(self.device)
        net_input = batch
        net_label = batch.y
        return net_input, net_label, batch.demo_id

    @staticmethod
    def get_new_features_args(generated_scenes, dataset=None):
        new_features_args = list()
        for i in range(generated_scenes.num_graphs):
            single_scene = generated_scenes.get_example(i)
            features = single_scene.x.detach().cpu().numpy()
            features_args = features[:2, :]
            new_features_args.append(features_args)
        return new_features_args

    def init_fixed_input(self, num_fixed, fixed_features_args, fixed_scene_id, dataset):
        fixed_scene = dataset.get_single_graph(
            fixed_scene_id, dataset.config.include_surrounding
        ).to(self.device)
        data_list = [fixed_scene] * num_fixed
        fixed_scenes = pyg.data.batch.Batch.from_data_list(data_list)
        return fixed_scenes

    def evaluate_gt_classifier(
        self, batch, clf, dataset, get_reason=False, demo_ids=None
    ):
        batch_tensor, mask = pyg.utils.to_dense_batch(batch.x, batch.batch)
        if get_reason:
            return clf.check_reason(batch_tensor, mask, demo_ids)
        else:
            return clf.check(batch_tensor, mask, demo_ids)

    def get_index_by_id(self, dataset, sample_id):
        return dataset.base_ds.demo_ids.index(sample_id)

    def get_manual_features_vis(self, batch, dataset):
        raise NotImplementedError


class DatasetAdapterStandard(DatasetAdapter):
    def __init__(self, device):
        super().__init__(device)

    def get_batch_size(self, batch):
        if type(batch) is list:
            return batch[0].shape[0]
        else:
            return batch.shape[0]

    def interpolate(self, real_sample, fake_sample):
        batch_size = self.get_batch_size(real_sample)
        eta = torch.rand(batch_size, 1, 1).to(self.device)
        eta = eta.expand_as(real_sample)
        interpolated = eta * real_sample + (1 - eta) * fake_sample
        interpolated.requires_grad_(True)
        return interpolated

    def compute_gradient_norm(
        self, in_interpolated, out_interpolated, label, grad_outputs
    ):
        batch_size = in_interpolated.shape[0]
        net_input = (in_interpolated, label) if label is not None else in_interpolated
        gradients = torch.autograd.grad(
            out_interpolated,
            net_input,
            grad_outputs=grad_outputs,
            create_graph=True,
            retain_graph=True,
            allow_unused=True,
        )[0]
        gradients = gradients.view(batch_size, -1)
        gradient_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)
        return gradient_norm

    def prepare_batch(self, batch):
        features, labels, demo_ids = batch
        features.to(self.device)
        labels.to(self.device)
        net_input = features
        net_label = labels
        return net_input, net_label, demo_ids

    @staticmethod
    def get_new_features_args(generated_scenes, dataset=None):
        generated_scenes_local = generated_scenes.detach().cpu().numpy()
        new_features_args = list()
        for idx in range(generated_scenes_local.shape[0]):
            new_features_args.append(generated_scenes_local[idx, :, :])
        return new_features_args

    def init_fixed_input(self, num_fixed, fixed_features_args, fixed_scene_id, dataset):
        return (
            torch.from_numpy(fixed_features_args)
            .float()
            .expand((num_fixed, fixed_features_args.shape[0], -1))
            .to(self.device)
        )

    def evaluate_gt_classifier(
        self, batch, clf, dataset, get_reason=False, demo_ids=None
    ):
        mask = torch.ones(batch.size(0), batch.size(1)).to(self.device)
        if get_reason:
            return clf.check_reason(batch, mask, demo_ids)
        else:
            return clf.check(batch, mask, demo_ids)

    def get_index_by_id(self, dataset, sample_id):
        return dataset.demo_ids.index(sample_id)

    def get_manual_features_vis(self, batch, dataset):
        raise NotImplementedError


def create_data_adapters(loaders, device):
    adapters = dict()
    for label in loaders:
        if type(loaders[label]["train"]) is pyg.loader.DataLoader:
            adapters[label] = DatasetAdapterGraph(device)
        elif type(loaders[label]["train"]) is TorchDataLoader:
            adapters[label] = DatasetAdapterStandard(device)
    return adapters


class TrainingSequenceGANBase:
    def __init__(
        self,
        run_config: tu.TrainingConfig,
        model_disc_container: tu.ModelSplitter,
        model_gen: nn.Module,
        device,
        out_dir,
        loaders,
        dataset,
        paths: Dict,
        resume: bool = False,
        debug: bool = False,
        ground_truth_classifier=None,
        existing_optimizer_filenames=None,
    ):
        self.model_disc_container = model_disc_container
        self.model_gen = model_gen
        self.device = device
        self.config = run_config
        self.loaders = loaders
        self.debug = debug
        self.dataset = dataset
        self.paths = paths
        self.ground_truth_classifier = ground_truth_classifier
        self.start_time = 0.0

        self.model_save_dir = os.path.join(out_dir, "models")
        os.makedirs(self.model_save_dir, exist_ok=True)

        # Data
        if not hasattr(self, "data_adapters"):
            self.data_adapters = create_data_adapters(loaders, self.device)

        # Optimizers
        params_disc = filter(
            lambda p: p.requires_grad, self.model_disc_container.mdl.parameters()
        )
        params_gen = filter(lambda p: p.requires_grad, self.model_gen.parameters())
        if self.config.optimizer == "adam":
            self.optimizer_disc = torch.optim.Adam(
                params_disc, lr=run_config.learning_rate
            )
            self.optimizer_gen = torch.optim.Adam(
                params_gen, lr=run_config.learning_rate
            )
        elif self.config.optimizer == "rmsprop":
            self.optimizer_disc = torch.optim.RMSprop(
                params_disc, lr=run_config.learning_rate
            )
            self.optimizer_gen = torch.optim.RMSprop(
                params_gen, lr=run_config.learning_rate
            )
        else:
            raise ValueError("Invalid optimizer selected")

        self.optimizer_filenames = (
            dict()
            if existing_optimizer_filenames is None
            else existing_optimizer_filenames
        )
        self.optimizer_filenames["gen"] = os.path.join(
            self.model_save_dir, "latest_gen_optimizer_state.pt"
        )
        self.optimizer_filenames["disc"] = os.path.join(
            self.model_save_dir, "latest_disc_optimizer_state.pt"
        )
        if resume:
            self.load_optimizer_state()

        # Loss
        self.loss = None
        self.clipper = None
        self.torch_bce_loss = nn.BCEWithLogitsLoss(reduction="none")
        self.init_loss()

        if run_config.loss_aggregation == "sum":
            self.loss_aggregation = torch.sum
        elif run_config.loss_aggregation == "mean":
            self.loss_aggregation = torch.mean
        else:
            raise NotImplementedError

        # Loggers
        self.loggers = {
            label: tu.TrainLogger(label, out_dir, run_config.loss_aggregation)
            for label in (
                "train_gen",
                "val_gen",
                "test_gen",
                "train_disc",
                "val_disc",
                "test_disc",
            )
        }
        self.loggers["resources"] = tu.RessourceLogger("resources", out_dir)
        self.mem_usage_resource_max = 0
        self.mem_ps_gb_max = 0.0
        self.mem_ps_percent_max = 0.0
        self.own_pid = os.getpid()

        # Visualization
        if self.debug:
            flags = util.parse_arguments(["--method", "gui"])
            self.debug_sim_server = SimServer(
                flags,
                LoggerStdout(),
                self.paths,
                data_session_id=self.config.dataset_id,
            )

        # Fixed latent
        if self.config.save_img_period > 0:
            self.camera_pos = (1.2, -1.7, 1.8)
            self.img_out_dir = os.path.join(out_dir, "samples")
            os.makedirs(self.img_out_dir, exist_ok=True)
            num_fixed = 9
            self.num_fixed_rows = 3
            assert num_fixed % self.num_fixed_rows == 0
            self.num_fixed_cols = int(num_fixed / self.num_fixed_rows)
            self.fixed_latent = self.model_gen.sample_noise(
                num_fixed, self.config.dim_noise
            ).to(self.device)

            if len(self.config.save_img_fixed_scene_id) > 0:
                fixed_scene_id = self.data_adapters["gen"].get_index_by_id(
                    self.dataset["train"], self.config.save_img_fixed_scene_id
                )
                (
                    fixed_features_args,
                    self.fixed_features_others,
                    fixed_label,
                    fixed_demo_id,
                    self.fixed_object_names,
                ) = self.dataset["train"].get_single_by_demo_id(
                    self.config.save_img_fixed_scene_id, use_tensors=False
                )
                assert fixed_label == 0, "A negative scene should be selected as fixed"
            else:
                fixed_scene_id = -1
                while True:
                    fixed_scene_id += 1
                    (
                        fixed_features_args,
                        self.fixed_features_others,
                        fixed_label,
                        fixed_demo_id,
                        self.fixed_object_names,
                    ) = self.dataset["train"].get_single(
                        fixed_scene_id, use_tensors=False
                    )

                    # Make sure we get a negative sample (i.e. predicate not fulfilled)
                    if fixed_label == 0:
                        break
            self.fixed_scene = self.data_adapters["gen"].init_fixed_input(
                num_fixed, fixed_features_args, fixed_scene_id, self.dataset["train"]
            )

            flags = util.parse_arguments(["--method", "direct"])
            self.fixed_sim_server = SimServer(
                flags, LoggerStdout(), self.paths, data_session_id=run_config.dataset_id
            )
            self.fixed_sim_server.pfm.restore_demonstration_outside(
                run_config.predicate_name,
                fixed_demo_id,
                scale=self.dataset["train"].base_ds.data_scaling,
            )

    def init_loss(self):
        if self.config.loss == "minimax":
            self.loss = self._bce_loss
        elif self.config.loss == "wasserstein":
            self.loss = self._wasserstein_loss
            self.init_clipper()
        elif self.config.loss == "mixed":
            self.loss = self._mixed_loss
            self.init_clipper()
        else:
            raise ValueError("Invalid loss selected")

    def init_clipper(self):
        if self.config.clipper_type == "clamp":
            self.clipper = WeightClipperClamp(
                -self.config.clipper_radius, self.config.clipper_radius
            )
        elif self.config.clipper_type == "norm":
            self.clipper = WeightClipperNorm(self.config.clipper_radius)
        elif self.config.clipper_type != "none":
            raise ValueError("Invalid clipper type")

    def train(self, i_start, start_time):
        logging.info("Starting training")
        self.start_time = start_time
        if self.config.training_style == "iterations":
            self.train_iterations(self.loaders, i_start, self.config.num_gen_iterations)
        elif self.config.training_style == "epochs":
            self.train_epochs(self.loaders, i_start)

    def train_iterations(self, loaders, i_gen_start, num_gen_iterations):
        scheduler_disc = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer_disc, T_max=num_gen_iterations
        )
        scheduler_gen = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer_gen, T_max=num_gen_iterations
        )
        train_loader_disc = self.infinite_batches(
            loaders["disc"]["train"], "disc", only_full_batches=True
        )
        train_loader_gen = self.infinite_batches(
            loaders["gen"]["train"], "gen", only_full_batches=True
        )
        self.set_requires_grad(disc=True, gen=True)
        for i_gen in range(i_gen_start + 1, i_gen_start + num_gen_iterations + 1):
            batch_gen = None
            disc_iterations = (
                100
                if i_gen - i_gen_start < 25 or i_gen % 500 == 0
                else self.config.disc_iters_per_gen_iter
            )
            for i_disc in range(disc_iterations):
                batch_disc, _ = next(train_loader_disc)
                batch_gen, _ = next(train_loader_gen)
                ret_disc = self.single_iteration_disc(batch_disc, batch_gen, train=True)
                self.log_iteration_disc(
                    "train", ret_disc, scheduler_disc.get_last_lr()[0]
                )
            ret_gen = self.single_iteration_gen(batch_gen, "train", train=True)
            self.log_iteration_gen("train", ret_gen, scheduler_gen.get_last_lr()[0])

            if tu.is_nth_epoch(
                i_gen, self.config.train_log_period, i_gen_start + num_gen_iterations
            ):
                self.loggers["train_disc"].write_epoch(i_gen)
                self.loggers["train_gen"].write_epoch(i_gen)

            # Evaluate
            if tu.is_nth_epoch(
                i_gen, self.config.eval_period, i_gen_start + num_gen_iterations
            ):
                self.evaluate(loaders, i_gen)

            self.save_models(i_gen, i_gen_start + num_gen_iterations)
            self.save_image(i_gen, i_gen_start + num_gen_iterations)

            if self.config.use_lr_scheduler:
                scheduler_disc.step()
                scheduler_gen.step()
        i_gen = i_gen_start + num_gen_iterations
        self.loggers["train_disc"].write_epoch(i_gen)
        self.loggers["train_gen"].write_epoch(i_gen)

    def train_epochs(self, loaders, epoch_start):
        schedulers = {
            "disc": torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer_disc, T_max=self.config.num_epochs
            ),
            "gen": torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer_gen, T_max=self.config.num_epochs
            ),
        }
        for epoch in range(epoch_start + 1, epoch_start + self.config.num_epochs + 1):
            self.one_epoch(
                train=True, split="train", loaders=loaders, schedulers=schedulers
            )
            self.loggers["train_disc"].write_epoch(epoch)
            self.loggers["train_gen"].write_epoch(epoch)
            if tu.is_nth_epoch(epoch, self.config.eval_period, self.config.num_epochs):
                self.evaluate(loaders, epoch)
            self.save_models(epoch, self.config.num_epochs)
            self.save_image(epoch, self.config.num_epochs)
        logging.info("Training done")

    def evaluate(self, loaders, iteration):
        for split in ["val", "test"]:
            self.one_epoch(train=False, split=split, loaders=loaders)
            self.loggers[f"{split}_disc"].write_epoch(iteration)
            self.loggers[f"{split}_gen"].write_epoch(iteration)
        self.log_memory_time(iteration)

    def log_memory_time(self, iteration):
        time_since_start = time.time() - self.start_time
        time_per_iteration = time_since_start / iteration
        entry = {
            "time_since_start": time_since_start,
            "time_per_iteration": time_per_iteration,
        }
        if self.config.use_tracemalloc:
            mem_tracemalloc = tracemalloc.get_traced_memory()
            entry["mem_tracemalloc_now"] = mem_tracemalloc[0]
            entry["mem_tracemalloc_max"] = mem_tracemalloc[1]

        # Use ps
        try:
            res = subprocess.run(
                ["ps", "-p", str(self.own_pid), "-o", "rss,pmem"],
                capture_output=True,
                text=True,
            ).stdout
            res = res.split("\n")[1]
            res = res.split(" ")
            res = [item for item in res if item != ""]
            mem_gb_current = float(res[0]) / 1000000.0
            mem_percent_current = float(res[1])
        except IndexError:
            mem_gb_current = -1
            mem_percent_current = -1
        self.mem_ps_gb_max = max(mem_gb_current, self.mem_ps_gb_max)
        self.mem_ps_percent_max = max(mem_percent_current, self.mem_ps_percent_max)
        entry["mem_ps_now_gb"] = mem_gb_current
        entry["mem_ps_max_gb"] = self.mem_ps_gb_max
        entry["mem_ps_now_percent"] = mem_percent_current
        entry["mem_ps_max_percent"] = self.mem_ps_percent_max

        self.loggers["resources"].log_entry(iteration, entry)

    def save_model(self, model_state_dict, optim_state_dict, label, epoch, num_epochs):
        model_filename = os.path.join(
            self.model_save_dir, f"{label}_model_{epoch:06d}.pt"
        )
        torch.save(model_state_dict, model_filename)
        if epoch == num_epochs:
            # Save last model under consistent name
            model_final_filename = os.path.join(
                self.model_save_dir, f"{label}_model_final.pt"
            )
            shutil.copy(model_filename, model_final_filename)
            torch.save(optim_state_dict, self.optimizer_filenames[label])

    def save_models(self, epoch, num_epochs):
        if self.config.save_model_period > 0 and tu.is_nth_epoch(
            epoch, self.config.save_model_period, num_epochs
        ):
            self.save_model(
                self.model_gen.state_dict(),
                self.optimizer_gen.state_dict(),
                "gen",
                epoch,
                num_epochs,
            )
            self.save_model(
                self.model_disc_container.mdl.state_dict(),
                self.optimizer_disc.state_dict(),
                "disc",
                epoch,
                num_epochs,
            )

    def save_image(self, epoch, num_epochs):
        if self.config.save_img_period > 0 and tu.is_nth_epoch(
            epoch, self.config.save_img_period, num_epochs
        ):
            # Generate
            new_features_args = self.generate_vis()
            fig, axs = plt.subplots(
                self.num_fixed_rows, self.num_fixed_cols, figsize=(20, 15)
            )
            fig.tight_layout()
            for idx in range(len(new_features_args)):
                all_features = (
                    np.concatenate((new_features_args[idx], self.fixed_features_others))
                    if len(self.fixed_features_others) > 0
                    else new_features_args[idx]
                )
                all_names = (
                    self.fixed_object_names["args"] + self.fixed_object_names["others"]
                )
                self.fixed_sim_server.visualize_features(
                    all_features, all_names, show_ground=False
                )
                arg_center = np.mean(all_features[:2, :3], axis=0)
                camera_pos = arg_center + np.array([1.2, 0.0, 1.3])
                img = self.fixed_sim_server.world.capture_image(
                    camera_pos=tuple(camera_pos), target_pos=tuple(arg_center)
                )
                plot_row = int(np.floor(idx / self.num_fixed_cols))
                plot_col = idx % self.num_fixed_cols
                axs[plot_row][plot_col].imshow(img)
                axs[plot_row][plot_col].axis("off")
                show_border = True
                if show_border:
                    auto_axis = axs[plot_row][plot_col].axis()
                    rec = plt.Rectangle(
                        (auto_axis[0] - 0.7, auto_axis[2] - 0.2),
                        (auto_axis[1] - auto_axis[0]) + 1,
                        (auto_axis[3] - auto_axis[2]) + 0.4,
                        fill=False,
                        lw=2,
                    )
                    rec = axs[plot_row][plot_col].add_patch(rec)
                    rec.set_clip_on(False)
            plt.subplots_adjust(wspace=0, hspace=0)
            filename = os.path.join(self.img_out_dir, f"samples_{epoch:06d}.png")
            # plt.show()
            fig.suptitle(f"Iteration {epoch}")
            fig.savefig(filename, bbox_inches="tight")
            plt.close(fig)

    def load_optimizer_state(self):
        gen_state_dict = torch.load(self.optimizer_filenames["gen"])
        disc_state_dict = torch.load(self.optimizer_filenames["disc"])
        self.optimizer_gen.load_state_dict(gen_state_dict)
        self.optimizer_disc.load_state_dict(disc_state_dict)

    def one_epoch(self, train: bool, split, loaders, schedulers=None):
        num_iterations = 0
        if train:
            lr_disc = schedulers["disc"].get_last_lr()[0]
            lr_gen = schedulers["gen"].get_last_lr()[0]
        else:
            lr_disc = 0.0
            lr_gen = 0.0

        it_loader_disc = self.infinite_batches(
            loaders["disc"][split], "disc", only_full_batches=True
        )
        it_loader_gen = self.infinite_batches(
            loaders["gen"][split], "gen", only_full_batches=True
        )

        disc_epoch_over = False
        while True:
            train_gen_this_time = (
                (num_iterations % self.config.disc_iters_per_gen_iter == 0)
                or not train
                or disc_epoch_over
            )
            num_iterations += 1

            batch_gen, gen_epoch_over = next(it_loader_gen)
            if not disc_epoch_over:
                batch_disc, disc_epoch_over = next(it_loader_disc)
                self.set_requires_grad(disc=True, gen=False)
                ret_disc = self.single_iteration_disc(
                    batch_disc, batch_gen, train=train
                )
                self.log_iteration_disc(split, ret_disc, lr_disc)

            if train_gen_this_time:
                self.set_requires_grad(disc=True, gen=True)
                ret_gen = self.single_iteration_gen(batch_gen, split, train=train)
                self.log_iteration_gen(split, ret_gen, lr_gen)

            if disc_epoch_over and gen_epoch_over:
                break

        if train and self.config.use_lr_scheduler:
            schedulers["disc"].step()
            schedulers["gen"].step()

    @staticmethod
    def _compute_loss(pred, true, loss_fcn):
        pred = pred.squeeze(-1) if pred.ndim > 1 else pred
        true = true.squeeze(-1) if true.ndim > 1 else true
        true = true.float()
        loss, label = loss_fcn(pred, true)
        return loss, label

    def _bce_loss(self, pred, true):
        loss = self.torch_bce_loss(pred, true)
        label = "bce"
        return loss, label

    @staticmethod
    def _wasserstein_loss(pred, true):
        loss1 = torch.mul(pred, (1 - true))
        loss2 = -torch.mul(pred, true)
        label = "ws"
        return (loss1 + loss2), label

    def _mixed_loss(self, pred, true):
        if pred.ndim > 1:
            ws_loss, ws_label = self._wasserstein_loss(pred[:, 0], true[:, 0])
            bce_loss, bce_label = self._bce_loss(pred[:, 1], true[:, 1])
            label = [ws_label, bce_label]
            loss = torch.stack((ws_loss, bce_loss))
        else:
            loss, label = self._wasserstein_loss(pred, true)
        return loss, label

    def _compute_gradient_penalty(self, real_sample, fake_sample):
        interpolated = self.data_adapters["disc"].interpolate(real_sample, fake_sample)

        # Forward pass
        label = interpolated.y if self.config.pass_label_to_disc else None
        out_interpolated, _, _ = self.discriminate(interpolated, label, target=None)
        grad_outputs = torch.ones_like(out_interpolated).to(self.device)

        gradient_norm = self.data_adapters["disc"].compute_gradient_norm(
            interpolated, out_interpolated, label, grad_outputs
        )
        gradient_penalty = self.config.gradient_penalty_lambda * (
            (gradient_norm - 1) ** 2
        )
        return gradient_penalty

    def generate(self, batch, batch_size, noise=None):
        if noise is None:
            noise = self.model_gen.sample_noise(batch_size, self.config.dim_noise).to(
                self.device
            )

        # desired_labels = torch.randint(0, 2, (batch_size, 1)).to(self.device)
        desired_labels = (
            torch.ones((batch_size, 1)).to(self.device)
            if self.config.pass_label_to_gen
            else None
        )
        generated_scene = self.model_gen.forward(
            batch, noise, desired_label=desired_labels
        )

        # if self.config.data_label_arg_objects:
        #     num_args = self.dataset.get_num_arguments()
        #     arg_labels = torch.cat(
        #         (
        #             torch.ones((num_args, 1)),
        #             torch.zeros((generated_scene.x.shape[0] - num_args, 1)),
        #         ),
        #         dim=0,
        #     )
        #     generated_scene.x = torch.cat((generated_scene.x, arg_labels), dim=1)

        return generated_scene, desired_labels

    def generate_vis(self):
        batch_size = self.data_adapters["gen"].get_batch_size(self.fixed_scene)
        generated_scenes, _ = self.generate(
            batch=self.fixed_scene, batch_size=batch_size, noise=self.fixed_latent
        )
        new_features_args = self.data_adapters["gen"].get_new_features_args(
            generated_scenes
        )
        return new_features_args

    def discriminate(self, batch, label, target):
        label_in = label if self.config.pass_label_to_disc else None
        disc_output = self.model_disc_container.mdl.forward(batch, label_in)
        disc_output = disc_output[:, self.model_disc_container.output_idx]
        disc_pred = torch.sigmoid(disc_output)
        return disc_output, target, disc_pred

    def single_iteration_disc(self, batch_disc, batch_gen, train=False):
        self.set_models_train_eval(train)

        batch_size_disc = self.data_adapters["disc"].get_batch_size(batch_disc)
        batch_size_gen = self.data_adapters["gen"].get_batch_size(batch_gen)

        real_target = torch.ones(batch_size_disc, 1).to(self.device)
        fake_target = torch.zeros(batch_size_gen, 1).to(self.device)

        batch_disc_in, batch_disc_label, _ = self.data_adapters["disc"].prepare_batch(
            batch_disc
        )
        batch_gen_in, _, _ = self.data_adapters["gen"].prepare_batch(batch_gen)

        if self.debug:
            self.debug_sim_server.pfm.restore_demonstration_outside(
                self.config.predicate_name,
                batch_disc_in.demo_id[0],
                scale=self.dataset["train"].base_ds.data_scaling,
            )
            self.debug_sim_server.scene.show_object_labels()

            # Visualize original dataset features
            features_args, features_others, debug_label, debug_demo_id, obj_names = self.dataset[
                "train"
            ].base_ds.get_single_by_demo_id(
                batch_disc_in.demo_id[0], use_tensors=False
            )

            all_features = np.concatenate((features_args, features_others))
            all_names = obj_names["args"] + obj_names["others"]
            self.debug_sim_server.visualize_features(
                all_features, all_names, show_ground=False
            )
            self.debug_sim_server.world.capture_image(
                show=True, camera_pos=self.camera_pos
            )

            # Visualize batch, converted to manual features
            recovered_features = self.data_adapters["gen"].get_manual_features_vis(
                batch_disc_in, self.dataset["train"]
            )
            all_features_recovered = recovered_features[0]
            self.debug_sim_server.visualize_features(
                all_features_recovered, all_names, show_ground=False
            )

        # Real
        disc_output_real, disc_target_real, disc_pred_real = self.discriminate(
            batch_disc_in, batch_disc_label, real_target
        )
        disc_loss_real, loss_label = self._compute_loss(
            disc_output_real, disc_target_real, self.loss
        )

        # Fake
        with torch.no_grad():
            generated_scene, generated_labels = self.generate(
                batch_gen_in, batch_size_gen
            )
        disc_output_fake, disc_target_fake, disc_pred_fake = self.discriminate(
            generated_scene.detach(), generated_labels, fake_target
        )
        disc_loss_fake, _ = self._compute_loss(
            disc_output_fake, disc_target_fake, self.loss
        )

        if self.debug:
            features_args = generated_scene[0, :, :].detach().cpu().numpy()
            all_features = np.concatenate((features_args, features_others))
            all_names = obj_names["args"] + obj_names["others"]
            self.debug_sim_server.visualize_features(
                all_features, all_names, show_ground=False
            )

        # Loss
        if self.config.gradient_penalty:
            gradient_penalty = self._compute_gradient_penalty(
                batch_disc_in, generated_scene
            )
        else:
            gradient_penalty = torch.zeros(batch_size_disc)
        disc_loss = disc_loss_real + disc_loss_fake + gradient_penalty
        disc_loss_total = self.loss_aggregation(disc_loss)
        if train:
            self.optimizer_disc.zero_grad()
            disc_loss_total.backward()
            self.optimizer_disc.step()
            if self.clipper is not None:
                self.model_disc_container.mdl.apply(self.clipper)

        loss = {
            "total": disc_loss_total.item(),
            "real": self.loss_aggregation(disc_loss_real).item(),
            "fake": self.loss_aggregation(disc_loss_fake).item(),
            "gradient": self.loss_aggregation(gradient_penalty).item(),
        }
        return (
            disc_target_fake,
            disc_pred_fake,
            disc_target_real,
            disc_pred_real,
            loss,
        )

    def single_iteration_gen(self, batch, split, train=False):
        self.set_models_train_eval(train)

        batch_in, _, _ = self.data_adapters["gen"].prepare_batch(batch)
        batch_size = self.data_adapters["gen"].get_batch_size(batch_in)
        real_target = torch.ones(batch_size, 1).to(self.device)

        generated_scene, generated_labels = self.generate(batch_in, batch_size)
        gen_output, gen_target, gen_pred = self.discriminate(
            generated_scene, generated_labels, real_target
        )
        gen_loss, _ = self._compute_loss(gen_output, gen_target, self.loss)

        # Evaluate with ground truth classifier
        ground_truth_performance = (
            self.data_adapters["gen"].evaluate_gt_classifier(
                generated_scene, self.ground_truth_classifier, self.dataset[split]
            )
            if self.ground_truth_classifier is not None
            else None
        )

        gen_loss_total = self.loss_aggregation(gen_loss)
        self.optimizer_gen.zero_grad()
        self.optimizer_disc.zero_grad()
        if train:
            gen_loss_total.backward()
            self.optimizer_gen.step()

        loss = {"total": gen_loss_total.item()}
        return gen_target, gen_pred, loss, batch_size, ground_truth_performance

    def set_models_train_eval(self, train: bool):
        if train:
            self.model_disc_container.mdl.train()
            self.model_gen.train()
        else:
            self.model_disc_container.mdl.eval()
            self.model_gen.eval()

    def set_requires_grad(self, disc, gen):
        if self.config.set_requires_grad:
            for p in self.model_disc_container.mdl.parameters():
                p.requires_grad = disc
            for p in self.model_gen.parameters():
                p.requires_grad = gen

    def log_iteration_disc(self, split, data, lr=0.0):
        (
            disc_target_fake,
            disc_pred_fake,
            disc_target_real,
            disc_pred_real,
            disc_loss,
        ) = data
        # if self.config.disc_predict_label:
        #     self.loggers[f"{split}_disc"].record_iteration(
        #         true=torch.cat((disc_target_fake[:, 1], disc_target_real[:, 1]))
        #         .detach()
        #         .cpu(),
        #         pred=torch.cat((disc_pred_fake[:, 1], disc_pred_real[:, 1]))
        #         .detach()
        #         .cpu(),
        #         loss=disc_loss,
        #         lr=lr,
        #         batch_size=disc_target_fake.shape[0],
        #     )
        # else:
        self.loggers[f"{split}_disc"].record_iteration(
            loss=disc_loss, lr=lr, batch_size=disc_target_fake.shape[0]
        )

    def log_iteration_gen(self, split, data, lr=0.0):
        (gen_target, gen_pred, gen_loss, batch_size, ground_truth) = data
        # if self.config.disc_predict_label:
        #     self.loggers[f"{split}_gen"].record_iteration(
        #         true=gen_target[:, 1].detach().cpu(),
        #         pred=gen_pred[:, 1].detach().cpu(),
        #         loss=gen_loss,
        #         lr=lr,
        #         batch_size=gen_target.shape[0],
        #     )
        # else:
        self.loggers[f"{split}_gen"].record_iteration(
            loss=gen_loss,
            lr=lr,
            batch_size=batch_size,
            ground_truth=ground_truth.detach().cpu(),
        )

    def infinite_batches(self, loader, task_str: str, only_full_batches=False):
        dataset_over = False
        while True:
            for batch in loader:
                if (
                    only_full_batches
                    and self.data_adapters[task_str].get_batch_size(batch)
                    != self.config.batch_size
                ):
                    break
                yield batch, dataset_over
                dataset_over = False
            dataset_over = True


class TrainingSequenceGANctat(TrainingSequenceGANBase):
    def __init__(
        self,
        run_config: tu.TrainingConfig,
        model_class_container: tu.ModelSplitter,
        model_disc_container: tu.ModelSplitter,
        model_gen: nn.Module,
        device,
        out_dir,
        loaders,
        paths: Dict,
        resume=False,
        debug=False,
        dataset=None,
        ground_truth_classifier=None,
    ):
        self.model_class_container = model_class_container

        # Classifier setup (since it's not taken care of by ancestors)
        params_class = filter(
            lambda p: p.requires_grad, model_class_container.mdl.parameters()
        )
        if run_config.optimizer == "adam":
            self.optimizer_class = torch.optim.Adam(
                params_class, lr=run_config.learning_rate
            )
        elif run_config.optimizer == "rmsprop":
            self.optimizer_class = torch.optim.RMSprop(
                params_class, lr=run_config.learning_rate
            )
        else:
            raise ValueError("Invalid optimizer selected")

        model_save_dir = os.path.join(out_dir, "models")
        optimizer_filenames = {
            "class": os.path.join(model_save_dir, "latest_class_optimizer_state.pt")
        }

        super().__init__(
            run_config,
            model_disc_container,
            model_gen,
            device,
            out_dir,
            loaders,
            dataset,
            paths,
            resume,
            debug,
            ground_truth_classifier,
            existing_optimizer_filenames=optimizer_filenames,
        )

        # Create loggers for classifier
        for split in ("train", "val", "test"):
            label = f"{split}_class"
            self.loggers[label] = tu.TrainLogger(
                label, out_dir, run_config.loss_aggregation
            )

    def load_optimizer_state(self):
        super().load_optimizer_state()
        class_state_dict = torch.load(self.optimizer_filenames["class"])
        self.optimizer_class.load_state_dict(class_state_dict)

    def train(self, i_start, start_time):
        overall_index = i_start
        self.start_time = start_time
        logging.info("Starting training")
        for training_step, num_iterations in self.config.training_schedule.items():
            if training_step == "ct":
                # Classifier training
                self.train_iterations_class(self.loaders, overall_index, num_iterations)
            elif training_step == "at":
                # GAN training
                self.train_iterations(self.loaders, overall_index, num_iterations)
            else:
                raise ValueError("Invalid training step identifier")
            overall_index += num_iterations

    def train_epochs(self, loaders, epoch_start):
        raise NotImplementedError

    def train_iterations_class(self, loaders, i_start, num_iterations):
        scheduler_class = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer_class, T_max=num_iterations
        )
        train_loader_class = self.infinite_batches(
            loaders["class"]["train"], "class", only_full_batches=False
        )
        self.model_class_container.mdl.train()
        for i_class in range(i_start + 1, i_start + num_iterations + 1):
            batch, _ = next(train_loader_class)
            ret_class = self.single_iteration_class(batch, "train", train=True)
            self.log_iteration_class(
                "train", ret_class, scheduler_class.get_last_lr()[0]
            )

            if tu.is_nth_epoch(
                i_class, self.config.train_log_period, i_start + num_iterations
            ):
                self.loggers["train_class"].write_epoch(i_class)

            # Evaluate
            if tu.is_nth_epoch(
                i_class, self.config.eval_period, i_start + num_iterations
            ):
                self.evaluate(loaders, i_class)

            self.save_models(i_class, i_start + num_iterations)
            # self.save_image(i_class, i_start + num_iterations)  # Not really necessary since generator is not changing

            if self.config.use_lr_scheduler:
                scheduler_class.step()
        i_class = i_start + num_iterations
        self.loggers["train_class"].write_epoch(i_class)

    def single_iteration_class(self, batch, split, train=False):
        self.set_models_train_eval(train)
        batch_in, label_in, demo_ids = self.data_adapters["class"].prepare_batch(batch)

        # w_cl_bef = list(self.model_class_container.mdl.parameters())[0].clone()

        if self.debug:
            self.debug_sim_server.pfm.restore_demonstration_outside(
                "on_supporting_ig", demo_ids[0]
            )

            features_args, features_others, debug_label, debug_demo_id, obj_names = self.dataset[
                split
            ].get_single_by_demo_id(
                demo_ids[0], use_tensors=False
            )
            # all_features = np.concatenate((features_args, features_others))
            # all_names = obj_names["args"] + obj_names["others"]

            # features_args = batch_in[0, :, :]
            all_features = features_args
            all_names = obj_names["args"]

            self.debug_sim_server.visualize_features(
                all_features,
                all_names,
                show_ground=False,
                remove_nonmentioned_objects=True,
            )
            # self.debug_sim_server.world.capture_image(show=True, camera_pos=self.camera_pos)

        class_output, class_pred = self.classify(batch_in)
        class_loss, _ = self._bce_loss(class_output, label_in)
        class_loss_total = self.loss_aggregation(class_loss)
        if train:
            self.optimizer_class.zero_grad()
            class_loss_total.backward()
            self.optimizer_class.step()

        # w_cl_after = list(self.model_class_container.mdl.parameters())[0].clone()
        # print(f"Weights unchanged: {torch.equal(w_cl_bef, w_cl_after)}")

        loss = {"total": class_loss_total.item()}
        return label_in, class_pred, loss

    def single_iteration_gen(self, batch, split, train=False):
        self.set_models_train_eval(train)

        # w_cl_bef = list(self.model_class_container.mdl.parameters())[0].clone()
        # w_gen_bef = list(self.model_gen.parameters())[0].clone()

        # Train generator
        batch_in, label_in, _ = self.data_adapters["gen"].prepare_batch(batch)
        batch_size = self.data_adapters["gen"].get_batch_size(batch_in)
        real_target = torch.ones(batch_size, 1).to(self.device)

        generated_scene, generated_labels = self.generate(batch_in, batch_size)

        # Discriminate
        disc_output, disc_target, disc_pred = self.discriminate(
            generated_scene, generated_labels, real_target
        )
        gen_loss_disc, _ = self._compute_loss(disc_output, disc_target, self.loss)

        # Classify
        class_output, class_pred = self.classify(generated_scene)
        class_target = torch.ones_like(label_in).float().to(self.device)
        class_loss, _ = self._bce_loss(class_output, class_target)

        # Evaluate with ground truth classifier
        with torch.no_grad():
            ground_truth_performance = (
                self.data_adapters["gen"].evaluate_gt_classifier(
                    generated_scene, self.ground_truth_classifier, self.dataset[split]
                )
                if self.ground_truth_classifier is not None
                else None
            )

        if self.config.gen_loss_components == "disc":
            gen_loss_total = gen_loss_disc
        elif self.config.gen_loss_components == "class":
            gen_loss_total = class_loss
        elif self.config.gen_loss_components == "both":
            gen_loss_total = class_loss + gen_loss_disc
        else:
            raise ValueError
        gen_loss_total = self.loss_aggregation(gen_loss_total)

        self.optimizer_gen.zero_grad()
        self.optimizer_disc.zero_grad()
        self.optimizer_class.zero_grad()
        if train:
            gen_loss_total.backward()
            self.optimizer_gen.step()

        # w_cl_after = list(self.model_class_container.mdl.parameters())[0].clone()
        # w_gen_after = list(self.model_gen.parameters())[0].clone()
        # print(
        #     f"Weights unchanged: {torch.equal(w_cl_bef, w_cl_after)} {torch.equal(w_gen_bef, w_gen_after)}"
        # )

        loss = {
            "total": gen_loss_total.item(),
            "disc": self.loss_aggregation(gen_loss_disc).item(),
            "class": self.loss_aggregation(class_loss).item(),
        }
        return class_target, class_pred, loss, batch_size, ground_truth_performance

    def classify(self, batch):
        class_output = self.model_class_container.mdl.forward(batch)
        class_output = class_output[:, self.model_class_container.output_idx]
        pred_score = torch.sigmoid(class_output)
        return class_output, pred_score

    def log_iteration_class(self, split, data, lr=0.0):
        (label, pred, loss) = data
        self.loggers[f"{split}_class"].record_iteration(
            true=label.detach().cpu(),
            pred=pred.detach().cpu(),
            loss=loss,
            lr=lr,
            batch_size=label.shape[0],
        )

    def log_iteration_gen(self, split, data, lr=0.0):
        (gen_target, gen_pred, gen_loss, batch_size, ground_truth) = data
        gt = ground_truth.detach().cpu() if ground_truth is not None else None
        if gen_target is not None and gen_pred is not None:
            self.loggers[f"{split}_gen"].record_iteration(
                true=gen_target.detach().cpu(),
                pred=gen_pred.detach().cpu(),
                loss=gen_loss,
                lr=lr,
                batch_size=batch_size,
                ground_truth=gt,
            )
        else:
            self.loggers[f"{split}_gen"].record_iteration(
                loss=gen_loss, lr=lr, batch_size=batch_size, ground_truth=gt
            )

    def set_models_train_eval(self, train):
        super().set_models_train_eval(train)
        if train:
            self.model_class_container.mdl.train()
        else:
            self.model_class_container.mdl.eval()

    def evaluate(self, loaders, iteration):
        for split in ["val", "test"]:
            self.one_epoch(train=False, split=split, loaders=loaders)
            self.loggers[f"{split}_disc"].write_epoch(iteration)
            self.loggers[f"{split}_gen"].write_epoch(iteration)
            self.loggers[f"{split}_class"].write_epoch(iteration)
        self.log_memory_time(iteration)

    def one_epoch(self, train: bool, split, loaders, schedulers=None):
        super().one_epoch(train, split, loaders, schedulers)

        if train:
            lr_class = schedulers["class"].get_last_lr()[0]
        else:
            lr_class = 0.0
        it_loader_class = self.infinite_batches(
            loaders["class"][split], "class", only_full_batches=False
        )
        while True:
            batch, epoch_over = next(it_loader_class)
            ret_class = self.single_iteration_class(batch, split, train=train)
            self.log_iteration_class(split, ret_class, lr_class)
            if epoch_over:
                break
        if train and self.config.use_lr_scheduler:
            schedulers["class"].step()

    def save_models(self, epoch, num_epochs):
        super().save_models(epoch, num_epochs)

        if self.config.save_model_period > 0 and tu.is_nth_epoch(
            epoch, self.config.save_model_period, num_epochs
        ):
            self.save_model(
                self.model_class_container.mdl.state_dict(),
                self.optimizer_class.state_dict(),
                "class",
                epoch,
                num_epochs,
            )
