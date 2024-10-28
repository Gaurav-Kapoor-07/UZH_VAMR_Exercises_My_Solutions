import numpy as np
import math

def pose_vector_to_transformation_matrix(pose_vec: np.ndarray) -> np.ndarray:
    """
    Converts a 6x1 pose vector into a 4x4 transformation matrix.

    Args:
        pose_vec: 6x1 vector representing the pose as [wx, wy, wz, tx, ty, tz]

    Returns:
        T: 4x4 transformation matrix
    """

    rot_vec = pose_vec[:3]

    rot_vec_mag = ((rot_vec[0] ** 2) + (rot_vec[1] ** 2) + (rot_vec[2] ** 2)) ** 0.5

    if rot_vec_mag == 0:
        return np.eye(4)

    k = rot_vec / rot_vec_mag

    kx = k[0]
    ky = k[1]
    kz = k[2]

    k_matrix = np.zeros((3, 3))

    k_matrix[0, 0] = 0.0
    k_matrix[0, 1] = -kz 
    k_matrix[0, 2] = ky
    k_matrix[1, 0] = kz
    k_matrix[1, 1] = 0.0
    k_matrix[1, 2] = -kx
    k_matrix[2, 0] = -ky
    k_matrix[2, 1] = kx
    k_matrix[2, 2] = 0.0

    R = np.eye(3) + math.sin(rot_vec_mag) * k_matrix + (1 - math.cos(rot_vec_mag)) * np.dot(k_matrix, k_matrix) 

    T = np.zeros((4, 4))

    for i in range(3):
        for j in range(3):
            T[i, j] = R[i, j]           

    T[0, 3] = pose_vec[3]
    T[1, 3] = pose_vec[4]
    T[2, 3] = pose_vec[5]
    T[3, 3] = 1.0
    
    return T
