import cv2
import matplotlib.pyplot as plt
import numpy as np
import time
from scipy.spatial.transform import Rotation

from estimate_pose_dlt import estimatePoseDLT
from reproject_points import reprojectPoints
from draw_camera import drawCamera
from plot_trajectory_3D import plotTrajectory3D

def main():
    append_path = "/home/gaurav07/computer-vision/exercise_02_updated/02_pnp - exercise"
    # Load 
    #    - an undistorted image
    #    - the camera matrix
    #    - detected corners
    image_idx = 1
    undist_img_path = append_path + "/data/images_undistorted/img_%04d.jpg" % image_idx
    undist_img = cv2.imread(undist_img_path, cv2.IMREAD_GRAYSCALE)

    K = np.loadtxt(append_path + "/data/K.txt")
    # print(K[0, :])
    p_W_corners = 0.01 * np.loadtxt(append_path + "/data/p_W_corners.txt", delimiter = ",")
    num_corners = p_W_corners.shape[0]
    # print(p_W_corners)

    # Load the 2D projected points that have been detected on the
    # undistorted image into an array
    # TODO: Your code here

    detected_corners_2d = np.loadtxt(append_path + "/data/detected_corners.txt")
    # print(detected_corners_2d[0, :])
    # print(detected_corners_2d.shape)

    pts_2d = np.empty((0, 2))
    #print(pts_2d)
    
    for i in range(0, detected_corners_2d.shape[1], 2):
        # print(i)
        u = detected_corners_2d[0, i]
        # print(u)
        v = detected_corners_2d[0, i + 1]
        uv_array = np.array([[u, v]])
        
        pts_2d = np.vstack((pts_2d, uv_array))
        
    # print(pts_2d)
    # print(pts_2d.shape)
        
    # Now that we have the 2D <-> 3D correspondances let's find the camera pose
    # with respect to the world using the DLT algorithm
    # TODO: Your code here
    M = estimatePoseDLT(p=pts_2d, P=p_W_corners, K=K) 
    # print(M)

    p_reproj = reprojectPoints(P=p_W_corners, M_tilde=M, K=K)
    # print(p_reproj)

    # Plot the original 2D points and the reprojected points on the image
    # TODO: Your code here
    
    plt.figure()
    plt.imshow(undist_img, cmap = "gray")
    plt.scatter(pts_2d[:,0], pts_2d[:,1], marker = 'o')
    plt.scatter(p_reproj[:,0], p_reproj[:,1], marker = '+')
    # plt.show()

    # Make a 3D plot containing the corner positions and a visualization
    # of the camera axis
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(p_W_corners[:,0], p_W_corners[:,1], p_W_corners[:,2])
    # plt.show()

    # Position of the camera given in the world frame
    # TODO: Your code here

    # M_full = np.vstack((M, np.array([0.0, 0.0, 0.0, 1.0]))) 
    # print(M_full)

    # M_full_inv = np.linalg.inv(M_full)
    # print(M_full_inv)

    # rotMat = M_full_inv[:3, :3]
    rotMat = M[:, :3].T
    # print(rotMat)
    # pos = M_full_inv[:3, 3]
    pos = - rotMat @ M[:, 3].reshape((3, 1))
    pos = pos.ravel()
    # print(pos.shape)
    # print(pos)
    drawCamera(ax, pos, rotMat, length_scale = 0.1, head_size = 10)
    plt.show()    


def main_video():
    append_path = "/home/gaurav07/computer-vision/exercise_02_updated/02_pnp - exercise"
    K = np.loadtxt(append_path + "/data/K.txt")
    p_W_corners = 0.01 * np.loadtxt(append_path + "/data/p_W_corners.txt", delimiter = ",")
    num_corners = p_W_corners.shape[0]

    all_pts_2d = np.loadtxt(append_path + "/data/detected_corners.txt")
    num_images = all_pts_2d.shape[0]
    translations = np.zeros((num_images, 3))
    quaternions = np.zeros((num_images, 4))
    
    for j in range(num_images):
        
        pts_2d = np.empty((0, 2))
        
        for i in range(0, all_pts_2d.shape[1], 2):
            u = all_pts_2d[j, i]
            v = all_pts_2d[j, i + 1]
            uv_array = np.array([[u, v]])
            
            pts_2d = np.vstack((pts_2d, uv_array))
            
        # print(pts_2d)

        M = estimatePoseDLT(p=pts_2d, P=p_W_corners, K=K) 
        # print(M)

        p_reproj = reprojectPoints(P=p_W_corners, M_tilde=M, K=K)
        # print(p_reproj)

        rotMat = M[:, :3].T

        r = Rotation.from_matrix(rotMat)
        quaternions[j, :] = r.as_quat()

        pos = - rotMat @ M[:, 3].reshape((3, 1))
        pos = pos.ravel()

        translations[j, :] = pos
    
    # print(quaternions)

    fps = 30
    filename = "../motion.avi"
    plotTrajectory3D(fps, filename, translations, quaternions, p_W_corners)
    


if __name__=="__main__":
    # main()
    main_video()
