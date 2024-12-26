import numpy as np

from utils import cross2Matrix

def linearTriangulation(p1, p2, M1, M2):
    """ Linear Triangulation
     Input:
      - p1 np.ndarray(3, N): homogeneous coordinates of points in image 1
      - p2 np.ndarray(3, N): homogeneous coordinates of points in image 2
      - M1 np.ndarray(3, 4): projection matrix corresponding to first image
      - M2 np.ndarray(3, 4): projection matrix corresponding to second image

     Output:
      - P np.ndarray(4, N): homogeneous coordinates of 3-D points
    """
    pass
    # TODO: Your code here

    A = np.zeros((6, 4, p1.shape[1]))

    P = np.zeros((4, p1.shape[1]))

    for i in range(p1.shape[1]): # assuming both p1 and p2 have the same shape
      p1_vec = p1[:, i]
      p2_vec = p2[:, i]
      A[:3, :, i] = cross2Matrix(p1_vec) @ M1
      A[3:, :, i] = cross2Matrix(p2_vec) @ M2

      U, s, VT = np.linalg.svd(A[:, :, i])

      V = np.transpose(VT)
      P_vector = V[:, V.shape[1] - 1] 

      P[:, i] = P_vector / P_vector[3]

    return P





