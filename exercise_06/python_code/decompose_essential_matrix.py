import numpy as np


def decomposeEssentialMatrix(E):
    """ Given an essential matrix, compute the camera motion, i.e.,  R and T such
     that E ~ T_x R
     
     Input:
       - E(3,3) : Essential matrix

     Output:
       - R(3,3,2) : the two possible rotations
       - u3(3,1)   : a vector with the translation information
    """
    pass
    # TODO: Your code here

    U, s, VT = np.linalg.svd(E)

    u3 = U[:, U.shape[1] - 1] 

    W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])

    R1 = U @ W @ VT

    det_R1 = np.linalg.det(R1)

    if det_R1 < 0:
        R1 *= -1

    R2 = U @ W.T @ VT

    det_R2 = np.linalg.det(R2)

    if det_R2 < 0:
        R2 *= -1

    R = np.zeros((3, 3, 2))
    R[:, :, 0] = R1
    R[:, :, 1] = R2
    
    return R, u3
