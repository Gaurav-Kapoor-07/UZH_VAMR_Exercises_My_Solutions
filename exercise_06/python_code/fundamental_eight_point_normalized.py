import numpy as np

from fundamental_eight_point import fundamentalEightPoint
from normalise_2D_pts import normalise2DPts

def fundamentalEightPointNormalized(p1, p2):
    """ Normalized Version of the 8 Point algorithm
     Input: point correspondences
      - p1 np.ndarray(3,N): homogeneous coordinates of 2-D points in image 1
      - p2 np.ndarray(3,N): homogeneous coordinates of 2-D points in image 2

     Output:
      - F np.ndarray(3,3) : fundamental matrix
    """
    pass
    # TODO: Your code here

    p1_normalized, T1 = normalise2DPts(p1)
    p2_normalized, T2 = normalise2DPts(p2)

    F_tilde = fundamentalEightPoint(p1_normalized, p2_normalized)

    F_unnormalized = T2.T @ F_tilde @ T1

    return F_unnormalized
