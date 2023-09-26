import pybullet as pb
from highlevel_planning_py.tools_pl.util import ObjectInfo
from highlevel_planning_py.sim.world import WorldPybullet
import numpy as np
from scipy.spatial.transform import Rotation as R
import os


def get_cupboard_info(base_dir, pos, orient, scale=0.6, force_fixed_base=False):
    urdf_file = "parsed_xacros/cupboard2.urdf"
    urdf = os.path.join(base_dir, urdf_file)

    world = WorldPybullet("direct", sleep=False)
    tmp_model = world.add_model(urdf, position=pos, orientation=orient, scale=scale)

    rot = R.from_quat(orient)
    yaw = rot.as_euler("xyz", degrees=False)
    nav_angle = yaw[2] + np.pi * 3.0 / 2.0

    drawer_joint_idx = list()
    handle_link_idx = list()
    for i in range(pb.getNumJoints(tmp_model.uid, physicsClientId=world.client_id)):
        info = pb.getJointInfo(tmp_model.uid, i, physicsClientId=world.client_id)
        joint_name = info[1] if type(info[1]) is str else info[1].decode("utf-8")
        # print(info)
        if "drawer_joint" in joint_name and len(joint_name) == 13:
            drawer_joint_idx.append(i)
        if "drawer_handle_dummy_joint" in joint_name:
            # handle_num = int(joint_name.split("drawer_handle_dummy_joint")[1])
            handle_link_idx.append(info[16])

    world.close()

    grasp_orient = R.from_euler("xzy", [180, 0, -45], degrees=True)
    return ObjectInfo(
        urdf_path_=urdf_file,
        urdf_relative_to_="asset_dir",
        init_pos_=np.array(pos),
        init_orient_=np.array(orient),
        init_scale_=scale,
        grasp_pos_={link: [np.array([0.0, 0.0, 0.0])] for link in handle_link_idx},
        grasp_orient_={link: [grasp_orient.as_quat()] for link in handle_link_idx},
        nav_angle_=nav_angle,
        nav_min_dist_=0.6,
        grasp_links_=handle_link_idx,
        joint_setting_=[
            {"jnt_idx": i, "mode": pb.VELOCITY_CONTROL, "force": 0.0}
            for i in drawer_joint_idx
        ],
        force_fixed_base_=force_fixed_base,
    )
