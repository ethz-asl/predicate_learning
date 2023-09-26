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
import os.path as osp
import pickle
import torch
import torch_geometric as pyg

import pybullet as pb
import numpy as np
from scipy.spatial.transform.rotation import Rotation

from highlevel_planning_py.tools_pl import util_pybullet

# from highlevel_planning_py.sim.scene_planning_2 import ScenePlanning2

# from highlevel_planning_py.sim.scene_planning_dw import ScenePlanningDW
# from highlevel_planning_py.sim.scene_pred import ScenePred
from highlevel_planning_py.sim.scene_base import SceneBase

from highlevel_planning_py.predicate_learning.demonstrations import (
    PredicateDemonstrationManager,
)
from highlevel_planning_py.predicate_learning.features import PredicateFeatureManager
from highlevel_planning_py.predicate_learning import data_utils
from highlevel_planning_py.sim.fake_perception import FakePerceptionPipeline
from highlevel_planning_py.tools_pl.exploration_tools import get_items_closeby
from highlevel_planning_py.predicate_learning.models_gnn import GANgeneratorGNN
from highlevel_planning_py.predicate_learning.models_mlp import GeneratorMLP
from highlevel_planning_py.predicate_learning.dataset import PredicateDatasetBase
import highlevel_planning_py.predicate_learning.visualization_utils as vu


class SimServer:
    def __init__(
        self,
        flags,
        logger,
        paths: Dict,
        data_session_id="220608-190256_demonstrations_features",
        feature_version="v1",
        silent=False,
    ):
        self.running = False

        self.flags = flags
        self.logger = logger
        self.assets_dir = paths["asset_dir"]
        self.data_dir = paths["data_dir"]
        self.paths = paths
        self.feature_version = feature_version

        self.pred_dir = osp.join(self.data_dir, "predicates")
        self.data_session_id = data_session_id

        # Load existing simulation data if desired
        # savedir = os.path.join(BASEDIR, "data", "sim")
        # objects, robot_mdl = util_pybullet.restore_pybullet_sim(savedir, args)

        # Load config file
        # cfg = ConfigYaml(os.path.join(BASEDIR, "config", "main.yaml"))

        self.is_open = False

        # Create world
        self.world, self.scene, self.fpp, self.pfm, self.pdm, self.dataset = (
            None,
            None,
            None,
            None,
            None,
            None,
        )
        self.scene_definition = SceneBase
        self.init()

        self.generator = None
        self.generator_meta_data = None
        self.generator_noise = None
        self.generator_training_config = None
        self._init_generator_meta_data()

        if not silent:
            self.logger.info(f"Using data session {data_session_id}")
            self.logger.info("Finished SimServer initialization")

    def _print(self, msg):
        self.logger.info(msg)

    def init(self):
        if self.world is not None:
            self.world.reset()
            self.scene.add_objects(force_load=True)
        else:
            self.scene, self.world = util_pybullet.setup_pybullet_world(
                self.scene_definition, self.paths, self.flags
            )
        self.scene.show_object_labels()

        # Setup perception
        self.fpp = FakePerceptionPipeline(
            logger=self.logger, pb_client_id=self.world.client_id, paths=self.paths
        )
        self.fpp.reset()
        self.fpp.populate_from_scene(self.scene)

        # Predicate learning

        self.pdm = PredicateDemonstrationManager(
            os.path.join(self.pred_dir, "data"),
            self.data_session_id,
            self.scene,
            self.fpp,
        )
        self.pfm = PredicateFeatureManager(
            os.path.join(self.pred_dir, "data"),
            self.data_session_id,
            self.world,
            self.scene,
            self.fpp,
            self.scene_definition,
            self.logger,
            self.paths,
            feature_version=self.feature_version,
        )

        self.is_open = True

    @staticmethod
    def _process_args(raw_args):
        arguments = raw_args.split(",")
        for i, a in enumerate(arguments):
            arguments[i] = a.strip()
        return arguments

    def _move_manual(self, name, translation, rotation):
        try:
            uid = self.scene.objects[name].model.uid
        except KeyError:
            return "Invalid object name"
        pos, orient_quat = pb.getBasePositionAndOrientation(
            uid, physicsClientId=self.world.client_id
        )
        orient = Rotation.from_quat(orient_quat)
        orient_euler = orient.as_euler("xyz", degrees=True)
        orient_euler += np.array(rotation)
        new_orient = Rotation.from_euler("xyz", orient_euler, degrees=True)
        new_orient_quat = new_orient.as_quat()
        new_pos = np.array(pos) + np.array(translation)
        pb.resetBasePositionAndOrientation(
            uid,
            new_pos.tolist(),
            new_orient_quat.tolist(),
            physicsClientId=self.world.client_id,
        )
        return

    def _move_manual_callback(self, msg):
        ret = self._move_manual(msg.target_name, msg.translation, msg.rotation)
        if ret is not None:
            self.logger.warning(ret)

    def _snapshot(self, cmd, pred_name, pred_args, label, other_data):
        pred_args = self._process_args(pred_args)
        success = True
        if cmd == 0:
            success = self.pdm.capture_demonstration(pred_name, pred_args, label)
            self.logger.info(f"Captured demonstration: {success}")
        elif cmd == 1:
            success = self.pfm.extract_demo_features(pred_name)
            self.logger.info(f"Extracted features: {success}")
        # elif cmd == 2:
        #     success = self.rdm.build_rules(req.pred_name)
        # elif cmd == 3:
        #     success = self.rdm.classify(req.pred_name, pred_args)
        elif cmd == 4:
            self._show_generated_state(pred_args)
        elif cmd == 5:
            self._sample_gen_noise()
        elif cmd == 6:
            self.scene.restore_initial_poses()
        elif cmd == 7:
            success = self.pfm.restore_demonstration_outside(pred_name, other_data)
            # self.scene.show_object_labels()
        elif cmd == 8:
            success = self.pfm.restore_demonstration_outside(
                pred_name, other_data, load_next=True
            )
            # self.scene.show_object_labels()
        elif cmd == 9:
            success = self._visualize_features_from_dataset(pred_name)
        else:
            success = False
        return success

    def _randomize_callback(self, msg):
        selectors = np.array([int(e) for e in msg.selectors])
        selectors = np.reshape(selectors, (3, 3))
        arguments = self._process_args(msg.arguments)
        ignore_objects = self._process_args(msg.ignore_objects)

        # Get closeby objects
        closeby_objects = get_items_closeby(
            arguments, self.scene.objects, self.world.client_id, distance_limit=1.0
        )
        # TODO make this work with objects that are not the base_link (e.g. drawers)
        # This means making get_items_closeby compatible with the fpp
        closeby_objects = [obj for obj in closeby_objects if obj not in ignore_objects]

        # Move objects
        previous_running = self.running
        self.running = False
        self._randomize(arguments[0], selectors[0, :])
        if len(arguments) > 1:
            for obj in arguments[1:]:
                self._randomize(obj, selectors[1, :])
        for obj in closeby_objects:
            self._randomize(obj, selectors[2, :])
        self.running = previous_running

    def _randomize(self, object_name, movement_spec):
        object_ids = self.fpp.object_info["object_ids_by_name"][object_name]
        max_tries = 10

        # Only update position and/or orientation if this is the base link
        if object_ids[1] == -1:
            pos, orient = pb.getBasePositionAndOrientation(
                object_ids[0], physicsClientId=self.world.client_id
            )

            move_success = False
            for i in range(max_tries):
                # New position
                delta_pos = np.zeros(3)
                if movement_spec[0] == 1:  # Small change
                    delta_pos[:2] = 0.05 * np.random.uniform(-1, 1, 2)
                    delta_pos[2] = np.random.uniform(-0.01, 0.1)
                elif movement_spec[0] == 2:  # Large change
                    delta_pos[:2] = 1.0 * np.random.uniform(-1, 1, 2)
                    delta_pos[2] = np.random.uniform(-0.1, 1.0)
                new_pos = pos + delta_pos

                # New orientation
                if movement_spec[1] == 1:  # Small change
                    delta_orient_magnitude = 5.0
                elif movement_spec[1] == 2:  # Large change
                    delta_orient_magnitude = 25.0
                else:  # No change
                    delta_orient_magnitude = 0.0
                delta_orient = delta_orient_magnitude * np.random.uniform(-1, 1, 3)
                delta_orient = Rotation.from_euler("xyz", delta_orient, degrees=True)
                new_orient = Rotation.from_quat(orient) * delta_orient
                new_orient = new_orient.as_quat()

                pb.resetBasePositionAndOrientation(
                    object_ids[0],
                    new_pos,
                    new_orient,
                    physicsClientId=self.world.client_id,
                )
                in_collision = self._check_in_collision(object_ids[0])
                if not in_collision:
                    self.logger.debug(
                        f"Set position and/or orientation of {object_name}"
                    )
                    move_success = True
                    break
            if not move_success:
                pb.resetBasePositionAndOrientation(
                    object_ids[0], pos, orient, physicsClientId=self.world.client_id
                )

        # New joint state
        if object_ids[1] != -1 and movement_spec[2] != 0:
            joint_info = pb.getJointInfo(
                object_ids[0], object_ids[1], physicsClientId=self.world.client_id
            )
            joint_state = pb.getJointState(
                object_ids[0], object_ids[1], physicsClientId=self.world.client_id
            )
            move_success = False
            for i in range(max_tries):
                delta_pos = None
                if joint_info[2] == pb.JOINT_REVOLUTE:
                    if movement_spec[2] == 1:
                        delta_pos = np.deg2rad(5.0) * np.random.uniform(-1, 1)
                    elif movement_spec[2] == 2:
                        delta_pos = np.deg2rad(25.0) * np.random.uniform(-1, 1)
                elif joint_info[2] == pb.JOINT_PRISMATIC:
                    if movement_spec[2] == 1:
                        delta_pos = 0.02 * np.random.uniform(-1, 1)
                    elif movement_spec[2] == 2:
                        delta_pos = 0.2 * np.random.uniform(-1, 1)
                if delta_pos is not None:
                    new_joint_pos = joint_state[0] + delta_pos
                    new_joint_pos = np.max(
                        (np.min((new_joint_pos, joint_info[9])), joint_info[8])
                    )
                    pb.resetJointState(
                        object_ids[0],
                        object_ids[1],
                        new_joint_pos,
                        physicsClientId=self.world.client_id,
                    )
                    if not self._check_in_collision(object_ids[0]):
                        self.logger.debug(f"Set parent joint position of {object_name}")
                        move_success = True
                        break
                else:
                    move_success = True
                    break
            if not move_success:
                pb.resetJointState(
                    object_ids[0],
                    object_ids[1],
                    joint_state[0],
                    physicsClientId=self.world.client_id,
                )
        self.logger.debug("-----------------------------------------")

    def _check_in_collision(self, object_uid):
        for other in self.scene.objects:
            other_uid = self.scene.objects[other].model.uid
            if other_uid == object_uid:
                continue
            res = pb.getClosestPoints(
                object_uid, other_uid, 0.01, physicsClientId=self.world.client_id
            )
            for point in res:
                if point[8] < 0:
                    return True
        return False

    def _print_info_callback(self, msg):
        self.pfm.print_info_all_predicates()

    def loop(self):
        if self.running:
            self.world.step_one()

    def _init_generator_meta_data(self):
        self.generator_run_dir = osp.join(
            self.pred_dir, "training", "211217_110935_gan"
        )
        if not osp.isdir(self.generator_run_dir):
            # self.logger.warning(
            #     f"Generator directory does not exist: {self.generator_run_dir}"
            # )
            return

        # Get meta data
        filename = osp.join(self.generator_run_dir, "parameters_generator.pkl")
        with open(filename, "rb") as f:
            self.generator_meta_data = pickle.load(f)
        filename = osp.join(self.generator_run_dir, "parameters_training.pkl")
        with open(filename, "rb") as f:
            self.generator_training_config = pickle.load(f)

        # Available models
        tmp = list(os.listdir(osp.join(self.generator_run_dir, "models")))
        self.generator_available_models = [t for t in tmp if "gen_model" in t]
        self.generator_available_models.sort()

    def _init_generator(self, statefile_name):
        if self.generator_training_config.model_type == "gnn":
            model = GANgeneratorGNN(self.generator_meta_data)
        elif self.generator_training_config.model_type == "mlp":
            model = GeneratorMLP(self.generator_meta_data)
        else:
            raise ValueError("Invalid model style")
        state_file = osp.join(self.generator_run_dir, "models", statefile_name)
        model.load_state_dict(torch.load(state_file))
        self.generator = model
        self._print(f"Initialized generator with weights {statefile_name}")

    def _sample_gen_noise(self):
        if self.generator_training_config.model_type == "gnn":
            self.generator_noise = torch.randn(self.generator_training_config.dim_noise)
        elif self.generator_training_config.model_type == "mlp":
            self.generator_noise = torch.randn(
                (1, self.generator_training_config.dim_noise)
            )
        else:
            raise ValueError("Invalid model style")
        self._print(f"Sampled new generator noise: {self.generator_noise}")

    def _show_generated_state(self, pred_args):
        if self.generator is None or self.generator_noise is None:
            self.logger.warning(
                "Generator and/or noise not initialized. Please do so first."
            )
            return

        features_args, features_others, object_names_others = self.pfm.extract_outer_features(
            pred_args
        )
        label = False

        # Normalize
        pos_arg0 = np.copy(features_args[0, :])
        if self.generator_training_config.data_normalization_method == "first_arg":
            ref = np.copy(features_args[0, :])
        elif self.generator_training_config.data_normalization_method == "scene_center":
            ref = data_utils.compute_scene_centroid(features_args, features_others)
        else:
            raise ValueError("Invalid normalization method selected")
        features_args = data_utils.normalize_features(
            features_args, ref, self.feature_version
        )
        features_others = data_utils.normalize_features(
            features_others, ref, self.feature_version
        )
        label = torch.tensor(label).float()
        features_args = torch.from_numpy(features_args).float()
        features_others = torch.from_numpy(features_others).float()

        if self.generator_training_config.model_type == "gnn":
            new_features_args = self.generate_gnn(features_args, features_others, label)
        elif self.generator_training_config.model_type == "mlp":
            new_features_args = self.generate_mlp(features_args)
        else:
            raise ValueError
        new_features_args = new_features_args.detach().numpy()

        # Apply sample to the scene
        ref2 = np.copy(new_features_args[0, :])
        new_features_args = data_utils.normalize_features(
            new_features_args, ref2, self.feature_version
        )
        new_features_args = data_utils.unnormalize_features(
            new_features_args, pos_arg0, self.feature_version
        )

        self.visualize_features(
            new_features_args, pred_args, remove_nonmentioned_objects=False
        )

    def generate_mlp(self, features_args):
        batch = torch.unsqueeze(features_args, 0)
        generated_scene = self.generator(batch, self.generator_noise)
        new_features_args = generated_scene[0]
        return new_features_args

    def generate_gnn(self, features_args, features_others, label):
        # Build graph
        graph = data_utils.build_graph(
            features_args, features_others, label=label, demo_id=None
        )
        batch = pyg.data.batch.Batch.from_data_list([graph])

        # Generate sample
        noise = self.generator_noise.repeat(graph.x.size(0), 1)
        desired_label = torch.ones((1, 1))
        generated_scene = self.generator(batch, noise, desired_label)

        num_args = features_args.shape[0]
        new_features_args = generated_scene.x[:num_args, :].detach().numpy()
        return new_features_args

    def visualize_features(
        self,
        features,
        object_names: list,
        show_ground=True,
        remove_nonmentioned_objects=True,
        show_bb=False,
    ):
        assert features.shape[0] == len(object_names)
        if remove_nonmentioned_objects:
            obj_to_del = set()
            for obj_name in self.scene.objects:
                if obj_name not in object_names:
                    self.scene.objects[obj_name].model.remove()
                    obj_to_del.add(obj_name)
            for obj_name in obj_to_del:
                del self.scene.objects[obj_name]
        pb.removeAllUserDebugItems(physicsClientId=self.world.client_id)
        self.scene.user_debug_text_ids.clear()
        for i, obj_name in enumerate(object_names):
            obj_ids = self.fpp.object_info["object_ids_by_name"][obj_name]
            obj_info = self.fpp.object_info["objects_by_id"][obj_ids]

            # Compute base position
            if self.feature_version == "v1":
                orient = Rotation.from_quat(features[i, 9:13])
            elif self.feature_version in ["v2", "v3"]:
                orient = Rotation.from_quat(features[i, 3:7])
            else:
                raise ValueError
            com_pos = obj_info.centroid
            base_pos = features[i, :3] - orient.apply(com_pos, inverse=False)

            # Move object
            uid = obj_ids[0]
            pb.resetBasePositionAndOrientation(
                uid, base_pos, orient.as_quat(), physicsClientId=self.world.client_id
            )

        # AABB
        if show_bb:
            if self.feature_version == "v1":
                vu.visualize_aabb(features[:, 3:9], self.world.client_id)
            elif self.feature_version in ["v2", "v3"]:
                vu.visualize_oabb(features, self.world.client_id)

        if show_ground:
            self.world.add_ground_plane()
        else:
            self.world.remove_ground_plane()

    def _visualize_features_from_dataset(self, pred_name):
        # self.pfm.restore_demonstration_outside(pred_name, demo_id, load_next=False)
        if self.dataset is None or pred_name != self.dataset.predicate_name:
            self.dataset = PredicateDatasetBase(
                os.path.join(self.pred_dir, "data"),
                self.data_session_id,
                pred_name,
                use_tensors=False,
                normalization_method="scene_center",
                feature_version=self.feature_version,
            )
        features_args, features_others, _, _, obj_names = self.dataset.get_single_by_demo_id(
            self.pfm.last_demo_id
        )
        if features_others.ndim > 1:
            all_features = np.concatenate((features_args, features_others))
            all_names = obj_names["args"] + obj_names["others"]
        else:
            all_features = features_args
            all_names = obj_names["args"]
        self.visualize_features(
            all_features,
            all_names,
            show_ground=False,
            remove_nonmentioned_objects=True,
            show_bb=True,
        )
        self.scene.show_object_labels()
        return True

    def close(self):
        if self.is_open:
            self.world.close()
        self.is_open = False
