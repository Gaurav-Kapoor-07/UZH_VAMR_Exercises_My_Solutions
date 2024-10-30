import numpy as np

def reprojectPoints(P, M_tilde, K):
    # Reproject 3D points given a projection matrix
    #
    # P         [n x 3] coordinates of the 3d points in the world frame
    # M_tilde   [3 x 4] projection matrix
    # K         [3 x 3] camera matrix
    #
    # Returns [n x 2] coordinates of the reprojected 2d points

    pass
    # TODO: Your code here

    p_projection_arr = np.empty((0, 3))

    for i in range(P.shape[0]):
        P_c = np.append(P[i, :], 1)
        P_c = P_c.reshape((4, 1))
        # print(P_c.shape)

        p_projection = K @ M_tilde @ P_c
        z_c = p_projection[2, 0]
        
        p_projection /= z_c

        p_projection = p_projection.reshape((1, 3))

        p_projection_arr = np.vstack((p_projection_arr, p_projection))

    p_projection_arr = p_projection_arr[:, :2]
    # print(p_projection_arr.shape)

    return p_projection_arr