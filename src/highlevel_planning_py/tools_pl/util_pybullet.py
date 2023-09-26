import os
from typing import Dict
import pickle

import pybullet as p

from highlevel_planning_py.sim.world import WorldPybullet


def save_pybullet_sim(args, savedir, scene, robot=None):
    robot_mdl = robot.model if robot is not None else None
    if not args.reuse_objects:
        if not os.path.isdir(savedir):
            os.makedirs(savedir)
        with open(os.path.join(savedir, "objects.pkl"), "wb") as output:
            pickle.dump((scene.objects, robot_mdl), output)
        p.saveBullet(os.path.join(savedir, "state.bullet"))


def restore_pybullet_sim(savedir, args):
    objects = None
    robot_mdl = None
    if args.reuse_objects:
        with open(os.path.join(savedir, "objects.pkl"), "rb") as pkl_file:
            objects, robot_mdl = pickle.load(pkl_file)
    return objects, robot_mdl


def setup_pybullet_world(scene_object, paths: Dict, args, savedir=None, objects=None):
    # Create world
    world = WorldPybullet(
        style=args.method,
        sleep=args.sleep,
        load_objects=not args.reuse_objects,
        savedir=savedir,
    )
    p.setAdditionalSearchPath(paths["asset_dir"], physicsClientId=world.client_id)

    scene = scene_object(world, paths, restored_objects=objects)

    return scene, world
