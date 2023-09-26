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
import pickle
import gzip
from functools import lru_cache
import pandas as pd
import numpy as np
from tqdm import tqdm
from copy import deepcopy

from scipy.spatial.transform.rotation import Rotation

import highlevel_planning_py.predicate_learning.visualization_utils as vu
from highlevel_planning_py.sim.world import WorldPybullet
from highlevel_planning_py.sim.fake_perception import FakePerceptionPipeline
from highlevel_planning_py.tools_pl.exploration_tools import get_items_closeby

# Try importing open3d. Will not work on Euler (also not needed there).
try:
    import open3d as o3d
except ModuleNotFoundError:
    pass


DEBUG = False


class PredicateFeatureManager:
    def __init__(
        self,
        data_dir,
        data_session_id,
        outer_world,
        outer_scene,
        outer_perception,
        inner_scene_definition,
        logger,
        paths,
        feature_version="v1",
    ):
        self.logger = logger
        self.paths = paths
        self.feature_version = feature_version
        self.assets_dir = paths["asset_dir"]
        self.data_dir = data_dir
        self.data_session_id = data_session_id
        self.outer_world = outer_world
        self.outer_scene = outer_scene
        self.outer_perception = outer_perception
        self.inner_scene_definition = inner_scene_definition

        self.last_demo_pred_name = None
        self.last_demo_list = None
        self.last_demo_id = None

        # if len(data_dir) > 0:
        #     self.print_info_all_predicates()

    @staticmethod
    def load_existing_features(filename):
        if os.path.isfile(filename):
            with open(filename, "rb") as f:
                features, relations, meta_data = pickle.load(f)
        else:
            features = pd.DataFrame()
            relations = dict()

            meta_data = dict()
            meta_data["demos_processed"] = set()
        return features, relations, meta_data

    def extract_demo_features(self, name: str, override_feature_version: str = ""):
        """
        Extracts features for all demonstrations that were previously saved to disk.
        """
        feature_dir = os.path.join(
            self.data_dir, name, "features", self.data_session_id
        )
        os.makedirs(feature_dir, exist_ok=True)
        demo_dir = os.path.join(
            self.data_dir, name, "demonstrations", self.data_session_id
        )

        feature_version = (
            override_feature_version
            if len(override_feature_version) > 0
            else self.feature_version
        )

        self.logger.info("Start extracting features")

        # Load existing data
        filename = os.path.join(feature_dir, f"{name}_{feature_version}.pkl")
        features, relations, meta_data = self.load_existing_features(filename)

        # Start simulator
        if DEBUG:
            world = WorldPybullet("gui", sleep=False, enable_gui=True)
        else:
            world = WorldPybullet("direct", sleep=False)
        scene = self.inner_scene_definition(world, self.paths, restored_objects=dict())
        perception = FakePerceptionPipeline(self.logger, world.client_id, self.paths)
        perception.populate_from_scene(scene)

        demo_ids = os.listdir(demo_dir)
        demo_ids.sort()
        for demo_id in tqdm(demo_ids):
            if not os.path.isdir(os.path.join(demo_dir, demo_id)):
                continue

            # Skip if this was already processed
            if demo_id in meta_data["demos_processed"]:
                continue
            else:
                assert demo_id not in relations

            # Clear caches
            self.get_features.cache_clear()

            # Load demo meta data
            meta_file_name = os.path.join(demo_dir, demo_id, "demo.pkl")
            with open(meta_file_name, "rb") as f:
                demo_meta_data = pickle.load(f)
            arguments = demo_meta_data[0]
            label = demo_meta_data[1]
            objects = demo_meta_data[2]

            # Save label
            relations[demo_id] = {
                "label": label,
                "arguments": list(),
                "others": list(),
                "argument_names": list(),
                "other_names": list(),
            }

            # Make sure the number of arguments is correct
            if "num_arguments" not in meta_data:
                meta_data["num_arguments"] = len(arguments)
            else:
                assert meta_data["num_arguments"] == len(arguments)

            # Populate simulation
            if objects != scene.objects:
                world.reset()
                scene.set_objects(objects)
                try:
                    scene.add_objects(force_load=True)
                except RuntimeError as e:
                    self.logger.warning(e)
                    self.logger.warning(f"Skipping this demonstration ({demo_id})")
                    continue
                perception.reset()
                perception.populate_from_scene(scene)
            simstate_file_name = os.path.join(demo_dir, demo_id, "state.bullet")
            world.restore_state_file(simstate_file_name)

            # Change table orientation here for debug purposes
            # tmp_uid, _ = perception.object_info["object_ids_by_name"]["table"]
            # tmp_orient = Rotation.from_euler("xz", (30, 25), degrees=True)
            # pb.resetBasePositionAndOrientation(
            #     tmp_uid, [3.0, 3.0, 1.0], tmp_orient.as_quat(), world.client_id
            # )

            # Compute features for arguments
            for arg_idx, arg in enumerate(arguments):
                feature_dict = self.get_features(arg, perception, feature_version)
                try:
                    new_index = features.index[-1] + 1
                except IndexError:
                    new_index = 0
                new_row = pd.DataFrame(feature_dict, index=[new_index])
                features = pd.concat([features, new_row])
                relations[demo_id]["arguments"].append(new_index)
                relations[demo_id]["argument_names"].append(arg)

            closeby_objects = get_items_closeby(
                arguments, scene.objects, world.client_id, distance_limit=1.0
            )
            for closeby_object in closeby_objects:
                feature_dict = self.get_features(
                    closeby_object, perception, feature_version
                )
                try:
                    new_index = features.index[-1] + 1
                except IndexError:
                    new_index = 0
                new_row = pd.DataFrame(feature_dict, index=[new_index])
                features = pd.concat([features, new_row])
                relations[demo_id]["others"].append(new_index)
                relations[demo_id]["other_names"].append(closeby_object)

            # Visualize bounding boxes
            if DEBUG:
                vis_idx = relations[demo_id]["arguments"] + relations[demo_id]["others"]
                vis_features = features.iloc[vis_idx, :].to_numpy()
                if feature_version == "v1":
                    vu.visualize_aabb(vis_features[:, 3:9], world.client_id)
                elif feature_version == "v2":
                    vu.visualize_oabb(vis_features, world.client_id)
                elif feature_version == "v3":
                    vu.visualize_oabb_surfaces(vis_features, world.client_id)
                else:
                    raise NotImplementedError(
                        "Visualization not implemented for this version"
                    )

            # Mark this demo as processed
            meta_data["demos_processed"].add(demo_id)

        if "num_features" not in meta_data:
            meta_data["num_features"] = features.shape[1]

        # Save data
        with open(filename, "wb") as f:
            pickle.dump((features, relations, meta_data), f)

        # Clean up
        world.close()

        return True

    def extract_outer_features(self, arguments):
        features_arguments = list()
        features_others = list()

        self.get_features.cache_clear()

        # Compute features for arguments
        for arg_idx, arg in enumerate(arguments):
            feature_dict = self.get_features(
                arg, self.outer_perception, self.feature_version
            )
            features_arguments.append(list(feature_dict.values()))

        closeby_objects = get_items_closeby(
            arguments,
            self.outer_scene.objects,
            self.outer_world.client_id,
            distance_limit=1.0,
        )
        for closeby_object in closeby_objects:
            feature_dict = self.get_features(
                closeby_object, self.outer_perception, self.feature_version
            )
            features_others.append(list(feature_dict.values()))

        features_arguments = np.array(features_arguments)
        features_others = np.array(features_others)

        return features_arguments, features_others, closeby_objects

    @lru_cache(maxsize=None)
    def get_features(self, obj_name, perception, feature_version):
        com, aabb, orientation, oabb, shifted_oabb, oabb_surface_centers = perception.get_object_info(
            obj_name, observed_only=False
        )

        if feature_version == "v1":
            features = (com, aabb, orientation, oabb)
            labels = list()
            labels.extend(["com_x", "com_y", "com_z"])
            labels.extend(["aabb_min_x", "aabb_min_y", "aabb_min_z"])
            labels.extend(["aabb_max_x", "aabb_max_y", "aabb_max_z"])
            labels.extend(["orient_x", "orient_y", "orient_z", "orient_w"])
            labels.extend(["oabb_min_x", "oabb_min_y", "oabb_min_z"])
            labels.extend(["oabb_max_x", "oabb_max_y", "oabb_max_z"])
        elif feature_version == "v2":
            features = (com, orientation, shifted_oabb, oabb)
            labels = list()
            labels.extend(["com_x", "com_y", "com_z"])
            labels.extend(["orient_x", "orient_y", "orient_z", "orient_w"])
            labels.extend(["shifted_oabb_x1", "shifted_oabb_y1", "shifted_oabb_z1"])
            labels.extend(["shifted_oabb_x2", "shifted_oabb_y2", "shifted_oabb_z2"])
            labels.extend(["shifted_oabb_x3", "shifted_oabb_y3", "shifted_oabb_z3"])
            labels.extend(["shifted_oabb_x4", "shifted_oabb_y4", "shifted_oabb_z4"])
            labels.extend(["oabb_min_x", "oabb_min_y", "oabb_min_z"])
            labels.extend(["oabb_max_x", "oabb_max_y", "oabb_max_z"])
        elif feature_version == "v3":
            features = (com, orientation, oabb_surface_centers, oabb)
            labels = list()
            labels.extend(["com_x", "com_y", "com_z"])
            labels.extend(["orient_x", "orient_y", "orient_z", "orient_w"])
            labels.extend(["oabb_face1_x", "oabb_face1_y", "oabb_face1_z"])
            labels.extend(["oabb_face2_x", "oabb_face2_y", "oabb_face2_z"])
            labels.extend(["oabb_face3_x", "oabb_face3_y", "oabb_face3_z"])
            labels.extend(["oabb_min_x", "oabb_min_y", "oabb_min_z"])
            labels.extend(["oabb_max_x", "oabb_max_y", "oabb_max_z"])
        else:
            raise ValueError("Invalid feature version")
        features = np.concatenate([np.squeeze(f.reshape((1, -1))) for f in features])
        assert len(features) == len(labels)
        feature_dict = {labels[i]: features[i] for i in range(len(features))}
        return feature_dict

    def print_info_all_predicates(self):
        self.logger.info("---- Existing demonstrations ---------------------")
        predicate_names = os.listdir(self.data_dir)
        predicate_names.sort()
        for pred_name in predicate_names:
            feature_dir = os.path.join(
                self.data_dir, pred_name, "features", self.data_session_id
            )
            if not os.path.isdir(feature_dir):
                continue
            for feature_file in os.listdir(feature_dir):
                if not feature_file.endswith(".pkl"):
                    continue
                with open(os.path.join(feature_dir, feature_file), "rb") as f:
                    # data, meta_data = pickle.load(f)
                    features, relations, meta_data = pickle.load(f)
                num_demos = len(relations)
                num_positive = 0
                for demo_info in relations.values():
                    if demo_info["label"]:
                        num_positive += 1
                self.logger.info(
                    f"Predicate name: {pred_name[:-4]}. "
                    f"Num total: {num_demos}. "
                    f"Num positive: {num_positive}."
                )
        self.logger.info("-------------------------------------------------")

    def restore_demonstration_outside(
        self, pred_name, demo_id, load_next=False, scale=None
    ):
        this_demo_dir = os.path.join(
            self.data_dir, pred_name, "demonstrations", self.data_session_id
        )

        if pred_name != self.last_demo_pred_name:
            assert not load_next
            self.last_demo_list = os.listdir(this_demo_dir)
            self.last_demo_list.sort()
            self.last_demo_id = None
            self.last_demo_pred_name = pred_name

        if load_next:
            assert self.last_demo_id is not None
            last_idx = self.last_demo_list.index(self.last_demo_id)
            this_idx = last_idx + 1
            while True:
                if this_idx >= len(self.last_demo_list):
                    self.logger.warn("Reached end of demo list. Abort.")
                    return False
                meta_file_name = os.path.join(
                    this_demo_dir, str(self.last_demo_list[this_idx]), "demo.pkl"
                )
                if os.path.isfile(meta_file_name):
                    break
                this_idx += 1
            selected_demo_id = self.last_demo_list[this_idx]

        else:
            meta_file_name = os.path.join(this_demo_dir, demo_id, "demo.pkl")
            selected_demo_id = demo_id
        self.last_demo_id = selected_demo_id

        with open(meta_file_name, "rb") as f:
            demo_meta_data = pickle.load(f)
        arguments = demo_meta_data[0]
        label = demo_meta_data[1]
        objects = demo_meta_data[2]
        comment = demo_meta_data[3] if len(demo_meta_data) == 4 else ""

        if scale is not None:
            for _, obj in objects.items():
                obj.scale *= scale

        if objects != self.outer_scene.objects:
            self.outer_world.reset()
            self.outer_scene.set_objects(objects)
            self.outer_scene.add_objects(force_load=True)
            self.outer_perception.reset()
            self.outer_perception.populate_from_scene(self.outer_scene)
        simstate_file_name = os.path.join(
            this_demo_dir, selected_demo_id, "state.bullet"
        )
        self.outer_world.restore_state_file(simstate_file_name)
        self.logger.info(
            f"Restoring demo {selected_demo_id}\nArguments: {arguments}. Label: {label}. Comment: {comment}."
        )
        self.logger.info("Done restoring.")
        return True

    @lru_cache(maxsize=None)
    def get_point_cloud(self, mesh_path, num_points):
        mesh = o3d.io.read_triangle_mesh(mesh_path)
        mesh.compute_vertex_normals()

        # if DEBUG:
        #     mesh_coords = o3d.geometry.TriangleMesh().create_coordinate_frame(size=1.0)
        #     o3d.visualization.draw_geometries([mesh, mesh_coords])

        pcd = mesh.sample_points_poisson_disk(
            number_of_points=num_points, init_factor=5
        )

        if DEBUG:
            mesh_coords = o3d.geometry.TriangleMesh().create_coordinate_frame(size=0.3)
            o3d.visualization.draw_geometries([pcd, mesh_coords])

        return pcd

    def process_point_cloud(
        self, obj_info, three_d_dir, features, obj_centroid, num_points
    ):
        if "ycb" in obj_info.urdf_path:
            mesh_path = os.path.join(
                self.paths[obj_info.urdf_relative_to],
                obj_info.urdf_path,
                "textured_simple.obj",
            )
        else:
            mesh_name = obj_info.urdf_path
            if "/" in obj_info.urdf_path:
                mesh_name = mesh_name.split("/")[-1]
            mesh_name = "".join(mesh_name.split(".")[:-1])
            mesh_name += ".obj"
            mesh_path = os.path.join(three_d_dir, mesh_name)
        assert os.path.isfile(mesh_path)

        cloud_raw = self.get_point_cloud(mesh_path, num_points)
        cloud = deepcopy(cloud_raw)

        # Scale
        cloud.scale(obj_info.scale, center=(0, 0, 0))

        # Rotations
        orient = features[9:13]  # [x,y,z,w]
        orient_other_convention = np.array(
            [orient[i] for i in [3, 0, 1, 2]]
        )  # [w,x,y,z]
        rotation = cloud.get_rotation_matrix_from_quaternion(orient_other_convention)
        cloud.rotate(rotation, center=(0, 0, 0))

        # Translation
        com = features[0:3]
        orient_r = Rotation.from_quat(orient)
        urdf_pos = com - orient_r.apply(obj_centroid, inverse=False)
        cloud.translate(urdf_pos)

        return cloud

    def extract_point_clouds(self, name: str, num_points: int):
        feature_dir = os.path.join(
            self.data_dir, name, "features", self.data_session_id
        )
        demo_dir = os.path.join(
            self.data_dir, name, "demonstrations", self.data_session_id
        )
        cloud_dir = os.path.join(
            self.data_dir,
            name,
            "pointclouds",
            self.data_session_id,
            f"num_points-{num_points}",
        )
        three_d_dir = os.path.join(self.data_dir, "3d_data")

        feature_version = "v1"

        os.makedirs(cloud_dir, exist_ok=True)

        # Load existing data
        filename = os.path.join(feature_dir, f"{name}_{feature_version}.pkl")
        features, relations, meta_data = self.load_existing_features(filename)

        # Start simulator
        if DEBUG:
            world = WorldPybullet("gui", sleep=False, enable_gui=True)
        else:
            world = WorldPybullet("direct", sleep=False)
        scene = self.inner_scene_definition(world, self.paths, restored_objects=dict())
        perception = FakePerceptionPipeline(self.logger, world.client_id, self.paths)

        demo_ids = os.listdir(demo_dir)
        demo_ids.sort()
        for demo_id in tqdm(demo_ids):
            if not os.path.isdir(os.path.join(demo_dir, demo_id)):
                continue

            if demo_id not in relations:
                print(
                    f"WARNING: features for demo {demo_id} not extracted yet. Skipping..."
                )
                continue

            cloud_filename = os.path.join(cloud_dir, f"{demo_id}.gz")
            if os.path.isfile(cloud_filename):
                continue

            # Load demo meta data
            meta_file_name = os.path.join(demo_dir, demo_id, "demo.pkl")
            with open(meta_file_name, "rb") as f:
                demo_meta_data = pickle.load(f)
            label = demo_meta_data[1]
            objects = demo_meta_data[2]

            # Populate simulation
            if objects != scene.objects:
                world.reset()
                scene.set_objects(objects)
                try:
                    scene.add_objects(force_load=True)
                except RuntimeError as e:
                    self.logger.warning(e)
                    self.logger.warning(f"Skipping this demonstration ({demo_id})")
                    continue
                perception.reset()
                perception.populate_from_scene(scene)
            if DEBUG:
                simstate_file_name = os.path.join(demo_dir, demo_id, "state.bullet")
                world.restore_state_file(simstate_file_name)

            arg_clouds = list()
            other_clouds = list()
            arg_clouds_np = list()
            other_clouds_np = list()

            for arg_i, arg_name in enumerate(relations[demo_id]["argument_names"]):
                obj_info = objects[arg_name]
                features_this_obj = features.iloc[
                    relations[demo_id]["arguments"][arg_i], :
                ]
                obj_centroid = perception.get_object_centroid(arg_name)
                cloud = self.process_point_cloud(
                    obj_info, three_d_dir, features_this_obj, obj_centroid, num_points
                )
                if DEBUG:
                    arg_clouds.append(cloud)
                cloud_np = np.asarray(cloud.points)
                arg_clouds_np.append(cloud_np)

            for other_i, other_name in enumerate(relations[demo_id]["other_names"]):
                obj_info = objects[other_name]
                features_this_obj = features.iloc[
                    relations[demo_id]["others"][other_i], :
                ]
                obj_centroid = perception.get_object_centroid(other_name)
                cloud = self.process_point_cloud(
                    obj_info, three_d_dir, features_this_obj, obj_centroid, num_points
                )
                if DEBUG:
                    other_clouds.append(cloud)
                cloud_np = np.asarray(cloud.points)
                other_clouds_np.append(cloud_np)

            if DEBUG:
                o3d.visualization.draw_geometries(arg_clouds + other_clouds)

            # Store clouds
            save_data = {
                "arg_clouds": arg_clouds_np,
                "other_clouds": other_clouds_np,
                "label": label,
            }
            with gzip.open(cloud_filename, "wb") as f:
                pickle.dump(save_data, f)

        world.close()

    def extract_single_pointclouds(self, name: str, num_points: int):
        pred_dir = os.path.join(self.data_dir, name)
        cloud_dir = os.path.join(
            pred_dir, "pointclouds", self.data_session_id, f"num_points-{num_points}"
        )
        cloud_single_dir = os.path.join(
            pred_dir, "pointclouds_individual", self.data_session_id
        )
        os.makedirs(cloud_single_dir, exist_ok=True)

        object_clouds_np = list()

        demo_ids = os.listdir(cloud_dir)
        demo_ids.sort()
        for demo_id in tqdm(demo_ids):
            # Read file
            demo_filename = os.path.join(cloud_dir, demo_id)
            with gzip.open(demo_filename, "rb") as f:
                data = pickle.load(f)

            object_clouds_np.extend(data["arg_clouds"])
            # To keep the dataset a bit smaller, only use arg clouds for now
            # object_clouds_np.extend(data["other_clouds"])

        # Store clouds
        cloud_filename = os.path.join(cloud_single_dir, f"point_clouds-{num_points}.gz")
        with gzip.open(cloud_filename, "wb") as f:
            pickle.dump(object_clouds_np, f)
