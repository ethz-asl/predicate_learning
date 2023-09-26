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
