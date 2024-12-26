import numpy as np

def normalise2DPts(pts):
    """  normalises 2D homogeneous points

     Function translates and normalises a set of 2D homogeneous points
     so that their centroid is at the origin and their mean distance from
     the origin is sqrt(2).

     Usage:   [pts_tilde, T] = normalise2dpts(pts)

     Argument:
       pts -  3xN array of 2D homogeneous coordinates

     Returns:
       pts_tilde -  3xN array of transformed 2D homogeneous coordinates.
       T         -  The 3x3 transformation matrix, pts_tilde = T*pts
    """
    pass
    # TODO: Your code here

    pts_euclidean = pts / pts[2, :] # divide the 1st and 2nd row by the 3rd row
    
    centroid = np.sum(pts_euclidean[:2, :], axis=1) / pts.shape[1] # centroid = mean

    centroid_reshape = centroid.reshape(2, 1)
    
    diff = pts_euclidean[:2, :] - centroid_reshape

    diff_sq = diff**2
    
    sum_diff_sq = np.sum(diff_sq, axis=0)

    sum_sum_diff_sq = np.sum(sum_diff_sq)

    sigma = np.sqrt(sum_sum_diff_sq / pts.shape[1]) # rms error
    
    sqrt_sum_diff_sq = np.sqrt(sum_diff_sq)

    # sigma = np.sum(sqrt_sum_diff_sq) / pts.shape[1] # mean error 

    s = np.sqrt(2) / sigma

    T = np.array([[s, 0, -s * centroid[0]], [0, s, -s * centroid[1]], [0, 0, 1]])

    pts_tilde = T @ pts_euclidean
    
    return pts_tilde, T



