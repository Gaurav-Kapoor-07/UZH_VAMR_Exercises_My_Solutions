import numpy as np

from linear_triangulation import linearTriangulation

def disambiguateRelativePose(Rots,u3,points0_h,points1_h,K1,K2):
    """ DISAMBIGUATERELATIVEPOSE- finds the correct relative camera pose (among
     four possible configurations) by returning the one that yields points
     lying in front of the image plane (with positive depth).

     Arguments:
       Rots -  3x3x2: the two possible rotations returned by decomposeEssentialMatrix
       u3   -  a 3x1 vector with the translation information returned by decomposeEssentialMatrix
       p1   -  3xN homogeneous coordinates of point correspondences in image 1
       p2   -  3xN homogeneous coordinates of point correspondences in image 2
       K1   -  3x3 calibration matrix for camera 1
       K2   -  3x3 calibration matrix for camera 2

     Returns:
       R -  3x3 the correct rotation matrix
       T -  3x1 the correct translation vector

       where [R|t] = T_C2_C1 = T_C2_W is a transformation that maps points
       from the world coordinate system (identical to the coordinate system of camera 1)
       to camera 2.
    """
    pass
    # TODO: Your code here

    R1 = Rots[:, :, 0]  
    R2 = Rots[:, :, 1] 

    u3_append = u3.reshape(3, 1)

    I3 = np.eye(3)
    zero_translation = np.zeros((3, 1))

    M1 = K1 @ np.hstack((I3, zero_translation))

    M2 = np.zeros((3, 4, 4))

    M2[:, :, 0] = K2 @ np.hstack((R1, u3_append))
    M2[:, :, 1] = K2 @ np.hstack((R2, u3_append))
    M2[:, :, 2] = K2 @ np.hstack((R1, -u3_append))
    M2[:, :, 3] = K2 @ np.hstack((R2, -u3_append))

    max_count_positive_depth = 0
    M2_return_indice = 0

    for i in range(4):
      P = linearTriangulation(points0_h, points1_h, M1, M2[:, :, i])

      Z = P[2, :]

      count_positive_depth = np.sum(Z > 0)

      if count_positive_depth > max_count_positive_depth:
         max_count_positive_depth = count_positive_depth
         M2_return_indice = i

    if M2_return_indice == 0:
       print("returning R1, u3")
       return R1, u3
    elif M2_return_indice == 1:
       print("returning R2, u3")
       return R2, u3
    elif M2_return_indice == 2:
       print("returning R1, -u3")
       return R1, -u3
    else:
       print("returning R2, -u3")
       return R2, -u3
    


