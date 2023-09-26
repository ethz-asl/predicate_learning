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
import pybullet as pb
from datetime import datetime


# To show dataframes:
# pd.options.display.max_columns=20
# pd.options.display.expand_frame_repr=False
# self.data["on"].iloc[:,:]


class PredicateDemonstrationManager:
    def __init__(self, data_dir, data_session_id, scene, perception):
        self.scene = scene
        self.perception = perception
        self._data = dict()
        self._meta_data = dict()
        self.data_dir = data_dir
        self.data_session_id = data_session_id

    def capture_demonstration(
        self, name: str, arguments: list, label: bool, comment: str = ""
    ):
        demo_dir = os.path.join(
            self.data_dir, name, "demonstrations", self.data_session_id
        )
        os.makedirs(demo_dir, exist_ok=True)

        # Assert that arguments actually exist
        for arg in arguments:
            assert arg in self.perception.object_info["object_ids_by_name"]

        # Create directory to save data
        time_now = datetime.now()
        time_string = time_now.strftime("%y%m%d-%H%M%S-%f")
        this_demo_dir = os.path.join(demo_dir, time_string)

        os.makedirs(this_demo_dir, exist_ok=False)

        meta_file = os.path.join(demo_dir, "_meta.pkl")
        if os.path.isfile(meta_file):
            with open(meta_file, "rb") as f:
                meta_data = pickle.load(f)
            assert meta_data["num_args"] == len(arguments)
        else:
            meta_data = {"num_args": len(arguments)}
            with open(meta_file, "wb") as f:
                pickle.dump(meta_data, f)

        # Save pickle
        save_data = (arguments, label, self.scene.objects, comment)
        with open(os.path.join(this_demo_dir, "demo.pkl"), "wb") as output:
            pickle.dump(save_data, output)

        # Save human readable data
        save_string = (
            f"Predicate name: {name}\nArguments: {arguments}\nHolds: {label}\n"
            f"Time reported: {time_string}\nComment: {comment}\n"
            f"Num Bodies: {pb.getNumBodies(physicsClientId=self.scene.world.client_id)}\n"
        )
        with open(os.path.join(this_demo_dir, "info.txt"), "w") as output:
            output.write(save_string)

        # Save bullet state
        pb.saveBullet(
            os.path.join(this_demo_dir, "state.bullet"),
            physicsClientId=self.scene.world.client_id,
        )

        return True
