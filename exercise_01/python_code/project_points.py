import numpy as np

from distort_points import distort_points


def project_points(points_3d: np.ndarray,
                   K: np.ndarray,
                   D: np.ndarray) -> np.ndarray:
    """
    Projects 3d points to the image plane, given the camera matrix,
    and distortion coefficients.

    Args:
        points_3d: 3d points (3xN)
        K: camera matrix (3x3)
        D: distortion coefficients (4x1)

    Returns:
        projected_points: 2d points (2xN)
    """
    pose_array_camera_frame = np.transpose(points_3d)
    # print(pose_array_camera_frame.shape[0])
    
    image_frame_array = np.empty((0, 3))

    for l in range(pose_array_camera_frame.shape[0]):
        pose_array_camera_frame_nth_row = pose_array_camera_frame[l, :3]
        # print(pose_array_camera_frame_nth_row)
        # print(pose_array_camera_frame_nth_row[2])
        image_frame = np.dot(K, pose_array_camera_frame_nth_row.reshape((3, 1))) / pose_array_camera_frame_nth_row[2]
        image_frame_array = np.append(image_frame_array, image_frame.reshape((1, 3)), axis=0) 

    image_frame_array_undistorted = image_frame_array[:, :2].transpose()
    
    image_frame_array_distorted = np.empty((2, 0))

    u_0 = K[0, 2]
    v_0 = K[1, 2]

    for m in range(image_frame_array_undistorted.shape[1]):
        u = image_frame_array_undistorted[0, m]
        # print(u)
        v = image_frame_array_undistorted[1, m]
        # print(v)

        r = ((u - u_0) ** 2 + (v - v_0) ** 2) ** 0.5
        
        [u_d, v_d] = (1 + (D[0] * (r ** 2)) + (D[1] * (r ** 4))) * np.array([u - u_0, v - v_0]) + np.array([u_0, v_0])
        
        image_frame_array_distorted = np.append(image_frame_array_distorted, np.array([u_d, v_d]).reshape((2, 1)), axis=1) 

    # print(image_frame_array_distorted)
    return image_frame_array_distorted
    # return image_frame_array[:, :2].transpose()
