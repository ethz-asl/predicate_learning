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
