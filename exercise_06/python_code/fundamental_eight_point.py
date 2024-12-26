import numpy as np

def fundamentalEightPoint(p1, p2):
    """ The 8-point algorithm for the estimation of the fundamental matrix F

     The eight-point algorithm for the fundamental matrix with a posteriori
     enforcement of the singularity constraint (det(F)=0).
     Does not include data normalization.

     Reference: "Multiple View Geometry" (Hartley & Zisserman 2000), Sect. 10.1 page 262.

     Input: point correspondences
      - p1 np.ndarray(3,N): homogeneous coordinates of 2-D points in image 1
      - p2 np.ndarray(3,N): homogeneous coordinates of 2-D points in image 2

     Output:
      - F np.ndarray(3,3) : fundamental matrix
    """
    pass
    # TODO: Your code here

    Q = np.zeros((p1.shape[1], 9))

    for i in range(p1.shape[1]):
      p1_vec = p1[:, i].reshape(3, 1)
      p2_vec = p2[:, i].reshape(3, 1)

      Q_vec = np.kron(p1_vec, p2_vec)
      Q[i, :] = Q_vec.T

    U, s, VT = np.linalg.svd(Q)

    V = np.transpose(VT)
    F_vec = V[:, V.shape[1] - 1]

    F = np.zeros((3, 3))
    
    for j in range(3):
      for k in range(3):
        F[k, j] = F_vec[3 * j + k]

    UF, sF, VTF = np.linalg.svd(F)

    sF[2] = 0

    F_correct = UF @ np.diag(sF) @ VTF

    return F_correct





