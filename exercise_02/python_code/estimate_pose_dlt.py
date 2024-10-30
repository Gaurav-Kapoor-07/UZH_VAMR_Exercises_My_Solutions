import numpy as np
import math

def estimatePoseDLT(p, P, K):
    # Estimates the pose of a camera using a set of 2D-3D correspondences
    # and a given camera matrix.
    # 
    # p  [n x 2] array containing the undistorted coordinates of the 2D points
    # P  [n x 3] array containing the 3D point positions
    # K  [3 x 3] camera matrix
    #
    # Returns a [3 x 4] projection matrix of the form 
    #           M_tilde = [R_tilde | alpha * t] 
    # where R is a rotation matrix. M_tilde encodes the transformation 
    # that maps points from the world frame to the camera frame

    pass

    # Convert 2D to normalized coordinates
    # TODO: Your code here

    inv_K = np.empty((0, 0))
    determinant = np.linalg.det(K)
    if determinant != 0:
        inv_K = np.linalg.inv(K)
        # print(inv_K)
    else:
        print("The camera matrix is not invertible.")
        return
    
    p_normalized_arr = np.empty((0, 3))
    for i in range(p.shape[0]):
        p_add = np.append(p[i, :], 1.0)
        p_add = p_add.reshape((3, 1))
        # print(p_add)
        p_normalized = np.dot(inv_K, p_add)
        # print(p_normalized)
        # print(p_normalized.shape)
        p_normalized_arr = np.vstack((p_normalized_arr, p_normalized.reshape(1, 3)))
        
    # print(p_normalized_arr)
    # print(p_normalized_arr.shape)
    # Build measurement matrix Q
    # TODO: Your code here

    Q = np.empty((0, 12))
    for j in range(p_normalized_arr.shape[0]):
        
        x_w = P[j, 0]
        # print(x_w)
        y_w = P[j, 1]
        z_w = P[j, 2]
        
        x_p = p_normalized_arr[j, 0]
        # print(x_p)
        y_p = p_normalized_arr[j, 1]
        
        first_row = np.array([x_w, y_w, z_w, 1.0, 0.0, 0.0, 0.0, 0.0, - x_p * x_w, - x_p * y_w, - x_p * z_w, - x_p])
        Q = np.vstack((Q, first_row))

        second_row = np.array([0.0, 0.0, 0.0, 0.0, x_w, y_w, z_w, 1.0, - y_p * x_w, - y_p * y_w, - y_p * z_w, - y_p])
        Q = np.vstack((Q, second_row))

    # print(Q.shape)

    # Solve for Q.M_tilde = 0 subject to the constraint ||M_tilde||=1
    # TODO: Your code here

    U, s, VT = np.linalg.svd(Q)

    V = np.transpose(VT)
    # print(V.shape)
    M_vector = V[:, V.shape[1] - 1]
    # print(M_vector.shape)

    z_check = M_vector[M_vector.shape[0] - 1]
    if (z_check < 0):
        M_vector = - M_vector

    # print(M_vector)
    
    # Extract [R | t] with the correct scale
    # TODO: Your code here

    M = np.zeros((3, 4))
    for k in range(3):
        for l in range(4):
            M[k, l] = M_vector[4 * k + l]

    # print(M)

    R = M[:, :3]
    # print(R)

    T = M[:, 3]
    # print(T.shape)

    # Find the closest orthogonal matrix to R
    # https://en.wikipedia.org/wiki/Orthogonal_Procrustes_problem
    # TODO: Your code here

    U_r, s_r, VT_r = np.linalg.svd(R)

    # print(U_r)
    # print(VT_r)

    R_correct = np.dot(U_r, VT_r)
    # print(R_correct.shape)

    det_R_correct= np.linalg.det(R_correct)
    # print(det_R_correct)
    eye_check = np.dot(np.transpose(R_correct), R_correct)
    # print(eye_check)

    if math.isclose(det_R_correct, 1.0, rel_tol=1e-3):
        print("Determinant of the corrected rotation matrix = 1.0")
        if np.allclose(eye_check, np.eye(eye_check.shape[0]), atol=1e-3, rtol=1.0):
            print("matrix product of rotation matrix and its transpose is identity") 
        else:
            print("matrix product of rotation matrix and its transpose is identity is not identity, returning")
            return
    else:
        print("Determinant of the corrected rotation matrix not equal to 1.0, returning")
        return
    
    # Normalization scheme using the Frobenius norm:
    # recover the unknown scale using the fact that R_tilde is a true rotation matrix
    # TODO: Your code here

    alpha = np.linalg.norm(R_correct, ord='fro') / np.linalg.norm(R, ord='fro')
    # print(alpha)

    T_correct = alpha * T
    T_correct = T_correct.reshape((3, 1))
    # print(T_correct.shape)
    
    # Build M_tilde with the corrected rotation and scale
    # TODO: Your code here

    M_correct = np.hstack((R_correct, T_correct))
    # print(M_correct)

    return M_correct
