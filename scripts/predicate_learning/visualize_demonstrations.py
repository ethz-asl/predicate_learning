import os
import pickle

import igibson
import pandas as pd
from highlevel_planning_py.sim.world import WorldPybullet
from highlevel_planning_py.sim.fake_perception import FakePerceptionPipeline

from highlevel_planning_py.tools_pl.exploration_tools import get_items_closeby
from highlevel_planning_py.predicate_learning.features import PredicateFeatureManager
from highlevel_planning_py.tools_pl.logger_stdout import LoggerStdout
from highlevel_planning_py.sim.scene_base import SceneBase
from highlevel_planning_py.predicate_learning.visualization_utils import visualize_oabb

SRCROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
PATHS = {
    "": "",
    "data_dir": os.path.join(os.path.expanduser("~"), "Data", "highlevel_planning"),
    "asset_dir": os.path.join(SRCROOT, "data", "models"),
    "igibson_dir": igibson.assets_path,
}


class DemoVisualizer(PredicateFeatureManager):
    def __init__(
        self, data_dir, data_session_id, inner_scene_definition, logger, paths
    ):
        super().__init__(
            data_dir,
            data_session_id,
            outer_world=None,
            outer_scene=None,
            outer_perception=None,
            inner_scene_definition=inner_scene_definition,
            logger=logger,
            paths=paths,
        )

    def visualize_demonstrations(
        self, name: str, draw_oabb: bool = False, specific_sample_type: str = None
    ):
        """
        Extracts features for all demonstrations that were previously saved to disk.
        """
        self.logger.info("Start visualizing demonstrations")
        demo_dir = os.path.join(
            self.data_dir, name, "demonstrations", self.data_session_id
        )

        # Start simulator
        world = WorldPybullet("gui", sleep=False)
        scene = self.inner_scene_definition(world, PATHS, restored_objects=dict())
        perception = FakePerceptionPipeline(self.logger, world.client_id, PATHS)
        perception.populate_from_scene(scene)

        demo_ids = sorted(os.listdir(demo_dir))

        for demo_id in demo_ids:
            if not os.path.isdir(os.path.join(demo_dir, demo_id)):
                continue

            # Clear caches
            self.get_features.cache_clear()

            # Load demo meta data
            meta_file_name = os.path.join(demo_dir, demo_id, "demo.pkl")
            with open(meta_file_name, "rb") as f:
                demo_meta_data = pickle.load(f)
            arguments, label, objects, sample_type = demo_meta_data

            if specific_sample_type is not None and sample_type != specific_sample_type:
                continue

            # Populate simulation
            if objects != scene.objects:
                world.reset()
                scene.set_objects(objects)
                scene.add_objects(force_load=True)
                perception.reset()
                perception.populate_from_scene(scene)
                scene.show_object_labels()
            simstate_file_name = os.path.join(demo_dir, demo_id, "state.bullet")
            world.restore_state_file(simstate_file_name)

            if draw_oabb:
                features_args = pd.DataFrame()
                features_others = pd.DataFrame()

                # Compute features for arguments
                for arg_idx, arg in enumerate(arguments):
                    feature_dict = self.get_features(arg, perception, "v2")
                    new_row = pd.DataFrame([feature_dict])
                    features_args = pd.concat(
                        (features_args, new_row), ignore_index=True
                    )

                closeby_objects = get_items_closeby(
                    arguments, scene.objects, world.client_id, distance_limit=1.0
                )
                for closeby_object in closeby_objects:
                    feature_dict = self.get_features(closeby_object, perception, "v1")
                    new_row = pd.DataFrame([feature_dict])
                    features_others = pd.concat(
                        (features_others, new_row), ignore_index=True
                    )

                visualize_oabb(features_args.to_numpy(), world.client_id)

            self.logger.info(
                f"Demo {demo_id}. Label: {label}, arguments: {arguments}, sample type: {sample_type}"
            )
            input("Press enter to continue")
        # Clean up
        world.close()
        return True


def main():
    logger = LoggerStdout()
    data_dir = os.path.join(PATHS["data_dir"], "predicates", "data")

    dataset_id = "230606-124817_demonstrations_features"
    viz = DemoVisualizer(data_dir, dataset_id, SceneBase, logger, PATHS)
    viz.visualize_demonstrations("inside_drawer")

    # dataset_id = "220831-175353_demonstrations_features"
    # viz = DemoVisualizer(data_dir, dataset_id, SceneBase, logger, PATHS)
    # viz.visualize_demonstrations(
    #     "on_clutter_test", specific_sample_type="obj_on_collision"
    # )


if __name__ == "__main__":
    main()
