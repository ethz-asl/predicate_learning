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
import shutil
import numpy as np
from tqdm import tqdm

TEST_SPLIT = 0.1


def main():
    original_dataset_dir = (
        "/home/fjulian/Data/latent_3d_points/data/shape_net_core_uniform_samples_2048"
    )
    split_dataset_dir = "/home/fjulian/Data/latent_3d_points/data/shape_net_core_uniform_samples_2048_split"

    train_root = os.path.join(split_dataset_dir, "train_val")
    os.makedirs(train_root, exist_ok=False)
    test_root = os.path.join(split_dataset_dir, "test")
    os.makedirs(test_root, exist_ok=False)

    test_only_classes = {
        "02773838": "bag",
        "02834778": "bicycle",
        "04074963": "remote_control",
        "03710193": "mailbox",
    }

    counts = {"train_val": 0, "test": 0}

    for dir_name in tqdm(os.listdir(original_dataset_dir)):
        src_dir = os.path.join(original_dataset_dir, dir_name)
        samples = os.listdir(src_dir)
        samples.sort()  # For consistency

        # Test only class
        if dir_name in test_only_classes:
            dest_dir = os.path.join(test_root, dir_name)
            shutil.copytree(src_dir, dest_dir)
            counts["test"] += len(samples)
            continue

        # How many samples?
        num_test = int(np.floor(len(samples) * TEST_SPLIT))
        samples_test = samples[:num_test]
        samples_train = samples[num_test:]

        os.makedirs(os.path.join(train_root, dir_name), exist_ok=False)
        os.makedirs(os.path.join(test_root, dir_name), exist_ok=False)
        for sample in samples_train:
            src_file = os.path.join(src_dir, sample)
            dest_file = os.path.join(train_root, dir_name, sample)
            shutil.copy2(src_file, dest_file)
        counts["train_val"] += len(samples_train)
        for sample in samples_test:
            src_file = os.path.join(src_dir, sample)
            dest_file = os.path.join(test_root, dir_name, sample)
            shutil.copy2(src_file, dest_file)
        counts["test"] += len(samples_test)

    # Write stats file
    stats_file = os.path.join(split_dataset_dir, "stats.txt")
    stats = {"num_samples": counts, "test_only_classes": test_only_classes}
    with open(stats_file, "w") as f:
        f.write(str(stats))


if __name__ == "__main__":
    main()
