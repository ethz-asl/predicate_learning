import numpy as np
import pybullet as pb

COLOR_MAGENTA = (243 / 255, 0 / 255, 255 / 255)


def get_corners_from_aabb(extreme_corners):
    all_corners = [
        [extreme_corners[ii, 0], extreme_corners[j, 1], extreme_corners[k, 2]]
        for ii in range(2)
        for j in range(2)
        for k in range(2)
    ]
    return all_corners


def draw_frame_from_all_corners(all_corners, client_id):
    connections = (
        (0, 1),
        (0, 2),
        (0, 4),
        (1, 3),
        (1, 5),
        (2, 3),
        (2, 6),
        (3, 7),
        (4, 5),
        (4, 6),
        (5, 7),
        (6, 7),
    )
    for conn in connections:
        pb.addUserDebugLine(
            all_corners[conn[0]],
            all_corners[conn[1]],
            COLOR_MAGENTA,
            4.0,
            0,
            physicsClientId=client_id,
        )


def visualize_aabb(aabbs, client_id):
    pb.removeAllUserDebugItems(physicsClientId=client_id)
    for i in range(aabbs.shape[0]):
        min_corner = aabbs[i, :3]
        max_corner = aabbs[i, 3:]
        extreme_corners = np.array([min_corner, max_corner])
        all_corners = get_corners_from_aabb(extreme_corners)
        draw_frame_from_all_corners(all_corners, client_id)


def visualize_oabb(features, client_id):
    pb.removeAllUserDebugItems(physicsClientId=client_id)
    for i in range(features.shape[0]):
        shifted_oabb_corners = features[i, 7:19]
        all_corners = np.reshape(shifted_oabb_corners, (-1, 3))
        connections = ((0, 1), (0, 2), (0, 3))
        for conn in connections:
            pb.addUserDebugLine(
                all_corners[conn[0]],
                all_corners[conn[1]],
                COLOR_MAGENTA,
                4.0,
                0,
                physicsClientId=client_id,
            )


def move_outward(com, point, distance):
    norm = np.linalg.norm(point - com)
    assert norm != 0
    return point + distance * (point - com) / norm


def visualize_oabb_surfaces(features, client_id):
    pb.removeAllUserDebugItems(physicsClientId=client_id)
    for i in range(features.shape[0]):
        com = features[i, :3]
        com_and_surface_centers = [
            com,
            move_outward(com, features[i, 7:10], 0.2),
            move_outward(com, features[i, 10:13], 0.2),
            move_outward(com, features[i, 13:16], 0.2),
        ]
        connections = ((0, 1), (0, 2), (0, 3))
        for conn in connections:
            pb.addUserDebugLine(
                com_and_surface_centers[conn[0]],
                com_and_surface_centers[conn[1]],
                COLOR_MAGENTA,
                4.0,
                0,
                physicsClientId=client_id,
            )
