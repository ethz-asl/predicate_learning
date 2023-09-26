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
import pandas as pd
from typing import List
from highlevel_planning_py.predicate_learning import training_utils as tu


def load_data(
    filter_str: list,
    scores_filename: str = "scores.csv",
    keep_order: bool = False,
    load_extra_parameters: List[str] = None,
):
    polybox_base = "/home/fjulian/Polybox/PhD/Publications/2022 Predicates"
    data_dirs = [
        "/home/fjulian/Data/highlevel_planning/predicates/training",
        polybox_base + "/data_gan/current",
        polybox_base + "/data_gan/archive",
        polybox_base + "/data_pc_gan/current",
        polybox_base + "/data_pc_gan/archive",
        polybox_base + "/data_dt/current",
    ]
    trainings = list()
    for data_dir in data_dirs:
        this_trainings = os.listdir(data_dir)
        this_trainings.sort()
        selected_trainings = list()
        because_of_filter = list()
        for f_id, f_str in enumerate(filter_str):
            new_trainings = [tr for tr in this_trainings if f_str in tr]
            selected_trainings.extend(new_trainings)
            because_of_filter.extend([f_id] * len(new_trainings))
        # selected_trainings = list(set(selected_trainings))  # Deduplicate
        trainings.extend(
            [
                (data_dir, tr, because_of_filter[i])
                for i, tr in enumerate(selected_trainings)
            ]
        )

    if keep_order:
        trainings.sort(key=lambda x: (x[2], x[1]))
    else:
        trainings.sort(key=lambda x: x[1])
    trainings = [(t[0], t[1]) for t in trainings]

    if len(trainings) == 0:
        raise ValueError(f"No trainings found for filter string {filter_str}")

    # Import data
    all_data = pd.DataFrame()
    for training in trainings:
        res_file = os.path.join(training[0], training[1], "post_eval", scores_filename)
        if os.path.isfile(res_file):
            this_data = pd.read_csv(res_file)
            if load_extra_parameters is not None:
                params = tu.load_parameters(
                    os.path.join(training[0], training[1]), "parameters_training"
                )
                for param in load_extra_parameters:
                    this_data[param] = getattr(params, param)
            all_data = pd.concat((all_data, this_data), ignore_index=True)
        # else:
        #     print(f"WARNING: No evaluations files for run {training[1]}")

    return all_data


def replace_labels(all_data, replace_labels_map: dict):
    training_id_translator = dict()
    if replace_labels_map is not None:
        for old_label in all_data["training_id"].unique():
            for partial_labels in replace_labels_map:
                if type(partial_labels) is tuple:
                    for partial_label in partial_labels:
                        if partial_label in old_label:
                            all_data.loc[
                                all_data["training_id"] == old_label, "training_id"
                            ] = replace_labels_map[partial_labels]
                            training_id_translator[old_label] = replace_labels_map[
                                partial_labels
                            ]
                            break
                else:
                    if partial_labels in old_label:
                        all_data.loc[
                            all_data["training_id"] == old_label, "training_id"
                        ] = replace_labels_map[partial_labels]
                        training_id_translator[old_label] = replace_labels_map[
                            partial_labels
                        ]
                        break
    return training_id_translator


def extract_filter_str(replace_labels_map: dict):
    filter_str = list()
    for partial_labels in replace_labels_map:
        if type(partial_labels) is tuple:
            filter_str.extend(list(partial_labels))
        else:
            assert type(partial_labels) is str
            filter_str.append(partial_labels)
    return filter_str


def replace_dataset_labels(all_data):
    all_data.loc[all_data["dataset"] == "on_clutter", "predicate"] = "on_clutter"
    all_data.loc[all_data["dataset"] == "on_clutter_test", "predicate"] = "on_clutter"
    all_data.loc[all_data["dataset"] == "inside_drawer", "predicate"] = "inside_drawer"
    all_data.loc[
        all_data["dataset"] == "inside_drawer_test", "predicate"
    ] = "inside_drawer"
    all_data.loc[:, "dataset"].replace("on_supporting_ig", "train", inplace=True)
    all_data.loc[:, "dataset"].replace("on_supporting_ig_test", "test", inplace=True)
    all_data.loc[:, "dataset"].replace("on_clutter", "train", inplace=True)
    all_data.loc[:, "dataset"].replace("on_clutter_test", "test", inplace=True)
    all_data.loc[:, "dataset"].replace("inside_drawer", "train", inplace=True)
    all_data.loc[:, "dataset"].replace("inside_drawer_test", "test", inplace=True)
