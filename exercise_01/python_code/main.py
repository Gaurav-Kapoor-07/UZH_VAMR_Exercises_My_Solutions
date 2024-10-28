import cv2
# import cv2.cv2
import matplotlib.pyplot as plt
import numpy as np
import time
import math
from pose_vector_to_transformation_matrix import \
    pose_vector_to_transformation_matrix
from project_points import project_points
from undistort_image import undistort_image
from undistort_image_vectorized import undistort_image_vectorized


def main():
    
    with open("/home/gaurav07/computer-vision/exercise_01/data/poses.txt", "r") as file:
        first_line = file.readline().strip()
        w_to_c_poses = [float(value) for value in first_line.split(" ")]
        np_w_to_c_poses = np.array(w_to_c_poses)
        # print(np_w_to_c_poses)

    T_w_to_c_poses = pose_vector_to_transformation_matrix(pose_vec=np_w_to_c_poses)

    # print(T_w_to_c_poses)

    # load camera poses

    # each row i of matrix 'poses' contains the transformations that transforms
    # points expressed in the world frame to
    # points expressed in the camera frame

    # TODO: Your code here

    # define 3D corner positions
    # [Nx3] matrix containing the corners of the checkerboard as 3D points
    # (X,Y,Z), expressed in the world coordinate system

    rows = 6 # y
    columns = 9 # x
    square_size = 4.0 / 100.0 # m

    corner_positions = np.zeros((rows * columns, 4))

    for i in range(rows):
        for j in range(columns):
            index = i * (columns) + j
            # print(index, end='\n')
            corner_positions[index, 0] = square_size * j
            corner_positions[index, 1] = square_size * i
            corner_positions[index, 2] = 0.0
            corner_positions[index, 3] = 1.0

    corner_positions_offset = np.zeros((rows * columns, 4))
    for o in range(rows):
        for p in range(columns):
            index = o * (columns) + p
            # print(index, end='\n')
            corner_positions_offset[index, 0] = square_size * p
            corner_positions_offset[index, 1] = square_size * o
            corner_positions_offset[index, 2] = -2 * square_size
            corner_positions_offset[index, 3] = 1.0

    # print(corner_positions_offset)
    
    # np.meshgrid()
 
    # TODO: Your code here

    # load camera intrinsics
    # TODO: Your code here

    # load one image with a given index
    # TODO: Your code here

    # img = cv2.imread("/home/gaurav07/computer-vision/exercise_01/data/images/img_0001.jpg")

    img = cv2.imread("/home/gaurav07/computer-vision/exercise_01/data/images/img_0001.jpg", cv2.IMREAD_GRAYSCALE)
    # print(img.shape)

    K = np.array([[420.506712, 0.0, 355.208298], [0.0, 420.610940, 250.336787], [0.0, 0.0, 1.0]])

    D = np.array([-1.6774e-06, 2.5847e-12])

    u_0 = K[0, 2]
    v_0 = K[1, 2]

    height, width = img.shape

    k1, k2 = D[0], D[1]
    
    undistorted_image = np.zeros([height, width])

    for x in range(width):
        for y in range(height):
            
            u = x
            v = y

            r = ((u - u_0) ** 2 + (v - v_0) ** 2) ** 0.5
            
            [u_d, v_d] = (1 + (D[0] * (r ** 2)) + (D[1] * (r ** 4))) * np.array([u - u_0, v - v_0]) + np.array([u_0, v_0])

            # cp_arr = np.array([[x, y]])

            # xp = cp_arr[:, 0] - u_0
            # yp = cp_arr[:, 1] - v_0

            # r2 = xp**2 + yp**2
            # xpp = u_0 + xp * (1 + k1*r2 + k2*r2**2)
            # ypp = v_0 + yp * (1 + k1*r2 + k2*r2**2)

            # x_d = np.stack([xpp, ypp], axis=-1)
            # print(x_d)
            # u_new, v_new = x_d[0, :]
            # print(u_d)
            
            # distorted_image = np.append(distorted_image, np.array([u_d, v_d]).reshape((2, 1)), axis=1)

            # img[x, y] = img[math.floor(u_d), math.floor(v_d)]

            # if (math.floor(u_new) >= 0) & (math.floor(u_new) < width) & (math.floor(v_new) >= 0) & (math.floor(v_new) < height):
            #     undistorted_image[y, x] = img[math.floor(v_new), math.floor(u_new)]

            # # nearest neighbour
            # if (math.floor(u_d) >= 0) & (math.floor(u_d) < width) & (math.floor(v_d) >= 0) & (math.floor(v_d) < height):
            #     undistorted_image[y, x] = img[math.floor(v_d), math.floor(u_d)] 

            # bilinear interpolation
            a = u_d - math.floor(u_d)
            b = v_d - math.floor(v_d)
            if (math.floor(u_d) >= 0) & (math.floor(u_d)+1 < width) & (math.floor(v_d) >= 0) & (math.floor(v_d)+1 < height):
                undistorted_image[y, x] = (1 - b) * ((1 - a) * img[math.floor(v_d), math.floor(u_d)] + a * img[math.floor(v_d), math.floor(u_d)+1]) + b * ((1 - a) * img[math.floor(v_d) + 1, math.floor(u_d)] + a * img[math.floor(v_d) + 1, math.floor(u_d) + 1])

    # print(distorted_image[0])

    # project the corners on the image
    # compute the 4x4 homogeneous transformation matrix that maps points
    # from the world to the camera coordinate frame

    pose_array_camera_frame = np.empty((0, 4))
    square_pose_array_camera_frame = np.empty((0, 4))
    
    for k in range(rows * columns):
        corner_positions_nth_row = corner_positions[k, :]
        corner_positions_reshape = corner_positions_nth_row.reshape((4, 1))

        corner_positions_offset_nth_row = corner_positions_offset[k, :]
        corner_positions_offset_reshape = corner_positions_offset_nth_row.reshape((4, 1))
        
        pose_vector_camera_frame = np.dot(T_w_to_c_poses, corner_positions_reshape)

        square_pose_vector_camera_frame = np.dot(T_w_to_c_poses, corner_positions_offset_reshape)
        # print(pose_vector_camera_frame)

        pose_array_camera_frame = np.append(pose_array_camera_frame, pose_vector_camera_frame.reshape((1, 4)), axis=0) 
        square_pose_array_camera_frame = np.append(square_pose_array_camera_frame, square_pose_vector_camera_frame.reshape((1, 4)), axis=0)

    # print(pose_array_camera_frame)

    pose_array_camera_frame_trim = pose_array_camera_frame[:, :3]
    # print(pose_array_camera_frame_trim)
    square_pose_array_camera_frame_trim = square_pose_array_camera_frame[:, :3]

    points_3d = np.transpose(pose_array_camera_frame_trim)
    # print(points_3d.shape)
    square_points_3d = np.transpose(square_pose_array_camera_frame_trim)

    # camera_matrix = np.array([[420.506712, 0.0, 355.208298], [0.0, 420.610940, 250.336787], [0.0, 0.0, 1.0]])
    # print(camera_matrix)

    # image_frame_array = np.empty((0, 3))
    # for l in range(rows * columns):
    #     pose_array_camera_frame_nth_row = pose_array_camera_frame[l, :3]
    #     # print(pose_array_camera_frame_nth_row)
    #     # print(pose_array_camera_frame_nth_row[2])
    #     image_frame = np.dot(camera_matrix, pose_array_camera_frame_nth_row.reshape((3, 1))) / pose_array_camera_frame_nth_row[2]
    #     image_frame_array = np.append(image_frame_array, image_frame.reshape((1, 3)), axis=0) 

    # print(image_frame_array)
    # print(image_frame_array.shape)

    image_frame_array = project_points(points_3d=points_3d, K=K, D=D)
    # print(image_frame_array)
    # square_image_frame_array = project_points(points_3d=square_points_3d, K=camera_matrix, D=np.array([-1.6774e-06, 2.5847e-12]))
    # print(square_image_frame_array)

    # plt.scatter(square_image_frame_array[0, :], square_image_frame_array[1, :], color='red', marker='o')
    
    # square_matrix_x = np.zeros((rows, columns))
    # square_matrix_y = np.zeros((rows, columns))

    # cube_matrix_x = np.zeros((rows, columns))
    # cube_matrix_y = np.zeros((rows, columns))

    # for m in range(rows):
        # for n in range(columns):
            # square_matrix_x[m, n] = image_frame_array[0, m * (columns) + n] 
            # square_matrix_y[m, n] = image_frame_array[1, m * (columns) + n] 

            # cube_matrix_x[m, n] = square_image_frame_array[0, m * (columns) + n]
            # cube_matrix_y[m, n] = square_image_frame_array[1, m * (columns) + n]
    
    # print(square_matrix_y)

    # plot_square_x = np.array([square_matrix_x[1, 3], square_matrix_x[1, 5], square_matrix_x[3, 5], square_matrix_x[3, 3], square_matrix_x[1, 3]])
    # plot_square_y = np.array([square_matrix_y[1, 3], square_matrix_y[1, 5], square_matrix_y[3, 5], square_matrix_y[3, 3], square_matrix_y[1, 3]])    

    # plt.plot(plot_square_x, plot_square_y, color='red', linewidth=2)               
    
    # print(plot_square_y)

    # plot_cube_x = np.array([cube_matrix_x[1, 3], cube_matrix_x[1, 5], cube_matrix_x[3, 5], cube_matrix_x[3, 3], cube_matrix_x[1, 3]])
    # plot_cube_y = np.array([cube_matrix_y[1, 3], cube_matrix_y[1, 5], cube_matrix_y[3, 5], cube_matrix_y[3, 3], cube_matrix_y[1, 3]])     

    # plt.plot(plot_cube_x, plot_cube_y, color='red', linewidth=2)

    # plot_connect_x_1 = np.array([square_matrix_x[1, 3], cube_matrix_x[1, 3]])
    # plot_connect_y_1 = np.array([square_matrix_y[1, 3], cube_matrix_y[1, 3]])     

    # plt.plot(plot_connect_x_1, plot_connect_y_1, color='red', linewidth=2)

    # plot_connect_x_2 = np.array([square_matrix_x[1, 5], cube_matrix_x[1, 5]])
    # plot_connect_y_2 = np.array([square_matrix_y[1, 5], cube_matrix_y[1, 5]])     

    # plt.plot(plot_connect_x_2, plot_connect_y_2, color='red', linewidth=2)

    # plot_connect_x_3 = np.array([square_matrix_x[3, 5], cube_matrix_x[3, 5]])
    # plot_connect_y_3 = np.array([square_matrix_y[3, 5], cube_matrix_y[3, 5]])     

    # plt.plot(plot_connect_x_3, plot_connect_y_3, color='red', linewidth=2)

    # plot_connect_x_4 = np.array([square_matrix_x[3, 3], cube_matrix_x[3, 3]])
    # plot_connect_y_4 = np.array([square_matrix_y[3, 3], cube_matrix_y[3, 3]])     

    # plt.plot(plot_connect_x_4, plot_connect_y_4, color='red', linewidth=2)

    # plt.scatter(image_frame_array[0, :], image_frame_array[1, :], color='red', marker='o')
    # plt.imshow(img)
    plt.imshow(undistorted_image, cmap='gray')
    plt.show()
    # transform 3d points from world to current camera pose
    # TODO: Your code here

    # undistort image with bilinear interpolation
    """ Remove this comment if you have completed the code until here
    start_t = time.time()
    img_undistorted = undistort_image(img, K, D, bilinear_interpolation=True)
    print('Undistortion with bilinear interpolation completed in {}'.format(
        time.time() - start_t))

    # vectorized undistortion without bilinear interpolation
    start_t = time.time()
    img_undistorted_vectorized = undistort_image_vectorized(img, K, D)
    print('Vectorized undistortion completed in {}'.format(
        time.time() - start_t))
    
    plt.clf()
    plt.close()
    fig, axs = plt.subplots(2)
    axs[0].imshow(img_undistorted, cmap='gray')
    axs[0].set_axis_off()
    axs[0].set_title('With bilinear interpolation')
    axs[1].imshow(img_undistorted_vectorized, cmap='gray')
    axs[1].set_axis_off()
    axs[1].set_title('Without bilinear interpolation')
    plt.show()
    """

    # calculate the cube points to then draw the image
    # TODO: Your code here
    
    # Plot the cube
    """ Remove this comment if you have completed the code until here
    plt.clf()
    plt.close()
    plt.imshow(img_undistorted, cmap='gray')

    lw = 3

    # base layer of the cube
    plt.plot(cube_pts[[1, 3, 7, 5, 1], 0],
             cube_pts[[1, 3, 7, 5, 1], 1],
             'r-',
             linewidth=lw)

    # top layer of the cube
    plt.plot(cube_pts[[0, 2, 6, 4, 0], 0],
             cube_pts[[0, 2, 6, 4, 0], 1],
             'r-',
             linewidth=lw)

    # vertical lines
    plt.plot(cube_pts[[0, 1], 0], cube_pts[[0, 1], 1], 'r-', linewidth=lw)
    plt.plot(cube_pts[[2, 3], 0], cube_pts[[2, 3], 1], 'r-', linewidth=lw)
    plt.plot(cube_pts[[4, 5], 0], cube_pts[[4, 5], 1], 'r-', linewidth=lw)
    plt.plot(cube_pts[[6, 7], 0], cube_pts[[6, 7], 1], 'r-', linewidth=lw)

    plt.show()
    """


if __name__ == "__main__":
    main()
