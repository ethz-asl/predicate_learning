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
import re
import subprocess
import json
from collections import OrderedDict
from datetime import datetime
from itertools import product
import pprint
import argparse
import shutil


DRY_RUN = False
USE_SLURM = True
SHORTEN_ID_STR = ["--init_class_with", "--encoder_id", "--classifier_id"]


def parse_json_spec(spec_file):
    # Parse JSON file
    with open(spec_file, "r") as f:
        spec = json.load(f, object_pairs_hook=OrderedDict)

    return (
        spec["script_name"],
        spec["general_args"],
        spec["grids"],
        spec["training_type"],
        spec["run_parameters"],
    )


def main(spec_file):
    assert spec_file.endswith(".json")
    script_name, general_args, grids, training_type, run_parameters = parse_json_spec(
        spec_file
    )

    all_configs = list()
    all_varied_param_idx = list()
    all_grid_keys = list()
    total_num_runs = 0
    for grid in grids:
        this_runs = tuple(product(*grid.values()))
        all_configs.append(this_runs)
        total_num_runs += len(this_runs)

        # Determine which parameters are varied
        varied_param_idx = list()
        grid_keys = tuple(grid.keys())
        for i in range(len(grid)):
            if len(grid[grid_keys[i]]) > 1 or "*" in grid_keys[i]:
                varied_param_idx.append(i)

        all_grid_keys.append(grid_keys)
        all_varied_param_idx.append(tuple(varied_param_idx))

    time_now = datetime.now()
    time_string = time_now.strftime("%y%m%d_%H%M%S")

    hostname = os.uname().nodename
    if hostname == "julian-thinkpad":
        script = os.path.expanduser(
            "~/Code/ros_manipulation_ws/src/high_level_planning_private/highlevel_planning/scripts"
        )
    else:
        script = os.path.expanduser(
            "~/Code/high_level_planning_private/highlevel_planning/scripts"
        )
    script = os.path.join(script, script_name)

    if USE_SLURM:
        batch_id_regex = re.compile(r"job (\d+)")
    else:
        batch_id_regex = re.compile(r"Job <(\d+)>")

    commands = list()
    batch_ids = list()
    params = list()
    out_dirs = list()
    run_idx = 0
    for i_grid, configs in enumerate(all_configs):
        for config in configs:
            run_idx += 1
            id_string = f"{time_string}_{run_idx:02}_{training_type}"
            for varied_idx in all_varied_param_idx[i_grid]:
                key = all_grid_keys[i_grid][varied_idx].strip("*")
                value = config[varied_idx]
                if key in SHORTEN_ID_STR:
                    value = "_".join(value.split("_")[:3])
                id_string += f"_{key}-{value}"

            cmd = ["python3", script, "--id_string", id_string]
            if hostname == "julian-thinkpad":
                cmd.extend(["--paths", "local"])
            else:
                cmd.extend(["--paths", "euler"])
            cmd.extend(general_args)
            if run_parameters["use_gpu"]:
                cmd.append("--gpu")
                cmd.append("true")
            cmd_params = list()
            for i_arg, arg_name in enumerate(all_grid_keys[i_grid]):
                if config[i_arg] == "":
                    # Skip if the parameter value is an empty string
                    continue
                cmd_params.append(f"{arg_name.strip('*')}")
                if config[i_arg] is not None:
                    cmd_params.append(f"{config[i_arg]}")
            cmd += cmd_params
            if hostname == "julian-thinkpad":
                print("==========================================================")
                print(f"Starting command {run_idx}/{total_num_runs}: {cmd}")
                print("----------------------------------------------------------")
                if not DRY_RUN:
                    try:
                        res = subprocess.run(cmd)
                    except subprocess.CalledProcessError:
                        print(f"Failed to run command: {cmd}")
                        continue
                batch_id = str(run_idx)
            else:
                if USE_SLURM:
                    cmd_prefix = [
                        "sbatch",
                        "-n",
                        "1",
                        f"--cpus-per-task={run_parameters['n_cpus']}",
                        f"--time={run_parameters['time_minutes']}",
                        f"--mem-per-cpu={run_parameters['memory_mb']}",
                        "--mail-type=TIME_LIMIT,FAIL",
                        f"--output={id_string}.log",
                    ]
                    if run_parameters["use_gpu"]:
                        cmd_prefix.append("--gpus=1")
                    cmd = cmd_prefix + [f'--wrap="{" ".join(cmd)}"']
                    cmd = " ".join(cmd)
                    arg_shell = True
                else:
                    cmd_prefix = [
                        "bsub",
                        "-n",
                        "1",
                        "-W",
                        f"{run_parameters['time_minutes']}",
                        "-R",
                        f"rusage[mem={run_parameters['memory_mb']}]",
                        "-o",
                        f"{id_string}.log",
                    ]
                    cmd = cmd_prefix + cmd
                    arg_shell = False
                if not DRY_RUN:
                    try:
                        res = subprocess.run(
                            cmd, capture_output=True, text=True, shell=arg_shell
                        )
                    except subprocess.CalledProcessError:
                        print(f"Failed to run command: {cmd}")
                        continue
                    # print(res.stdout)
                    # print(res.stderr)
                    try:
                        batch_id = batch_id_regex.findall(res.stdout)[0]
                    except IndexError:
                        print("Failed to get batch ID.")
                        batch_id = str(run_idx)
                else:
                    batch_id = str(run_idx)
                    print(f"Command: {cmd}")
                print(f"Submitted job {run_idx}/{total_num_runs}")
            batch_ids.append(batch_id)
            commands.append(cmd)
            params.append(cmd_params)
            out_dirs.append(id_string)

    filename = f"{time_string}_grid.txt"
    with open(filename, "w") as f:
        f.write(f"Hostname: {hostname}\n")
        f.write("Grids:\n")
        pprint.pprint(grids, f, sort_dicts=False)
        f.write("\n")
        for i, batch_id in enumerate(batch_ids):
            f.write(f"Out dir: {out_dirs[i]}\n")
            f.write(f"Batch ID: {batch_id}\n")
            f.write(f"Params: {params[i]}\n")
            f.write(f"Command: {commands[i]}\n")
            f.write("\n")

    # Copy spec file
    description_str = ".".join(spec_file.split(".")[:-1])
    shutil.copy2(spec_file, f"{time_string}_spec_{description_str}.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("spec_file", type=str)
    args = parser.parse_args()
    main(args.spec_file)
