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


def get_path_dict(machine: str):
    src_root = os.path.join(os.path.dirname(__file__), "..", "..", "..")
    if machine == "local":
        return {
            "": "",
            "data_dir": os.path.join(src_root, "training"),
            "asset_dir": os.path.join(src_root, "data", "models"),
            "igibson_dir": os.path.join(
                os.path.expanduser("~"), "Data", "igibson", "assets"
            ),
        }
    elif machine == "home":
        return {
            "": "",
            "data_dir": os.path.join(
                os.path.expanduser("~"), "Data", "highlevel_planning"
            ),
            "asset_dir": os.path.join(src_root, "data", "models"),
            "igibson_dir": os.path.join(
                os.path.expanduser("~"), "Data", "igibson", "assets"
            ),
        }
    elif machine == "euler":
        return {
            "": "",
            "data_dir": "/cluster/work/riner/users/fjulian/highlevel_planning",
            "asset_dir": os.path.join(src_root, "data", "models"),
            "igibson_dir": "/cluster/work/riner/users/fjulian/igibson/assets",
        }
    else:
        raise ValueError(f"Unknown machine {machine}")
