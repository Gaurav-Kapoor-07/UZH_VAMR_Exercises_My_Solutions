import numpy as np


def parabolaRansac(data, max_noise):
    """
    best_guess_history is 3xnum_iterations with the polynome coefficients
    from polyfit of the BEST GUESS SO FAR at each iteration columnwise and
    max_num_inliers_history is 1xnum_iterations, with the inlier count of the
    BEST GUESS SO FAR at each iteration.
    """
    pass
    # TODO: Your code here

    num_iterations = 100

    best_guess_history = np.zeros((3, num_iterations))
    max_num_inliers_history = np.zeros(num_iterations)

    max_inlier_count = 0
    
    for j in range(num_iterations):
     
        indices = np.random.permutation(data.shape[1])[:3]
        cols = data[:, indices]

        rand_point_1 = cols[:, 0]
        rand_point_2 = cols[:, 1]
        rand_point_3 = cols[:, 2]

        rand_point_1_x = rand_point_1[0]
        rand_point_1_y = rand_point_1[1]

        rand_point_2_x = rand_point_2[0]
        rand_point_2_y = rand_point_2[1]

        rand_point_3_x = rand_point_3[0]
        rand_point_3_y = rand_point_3[1]

        # parabola_check = np.polyfit([rand_point_1_x, rand_point_2_x, rand_point_3_x], [rand_point_1_y, rand_point_2_y, rand_point_3_y], 2)
        # print("parabola_check =", parabola_check)

        # print(cols[0], rand_point_1[0], rand_point_1[1])
        # print()
        # print(cols[1], rand_point_2[0], rand_point_2[1])
        # print()
        # print(cols[2], rand_point_3[0], rand_point_3[1])

        # Create the matrix for solving the parabola
        A = np.array([[rand_point_1_x**2, rand_point_1_x, 1],
                    [rand_point_2_x**2, rand_point_2_x, 1],
                    [rand_point_3_x**2, rand_point_3_x, 1]])
        
        y = np.array([rand_point_1_y, rand_point_2_y, rand_point_3_y])

        inv_A = np.linalg.inv(A)

        parabola = inv_A @ y

        # print("parabola =", parabola)

        count = 0
        dist = np.zeros(data.shape[1])
        # print(dist)

        data_inlier = []
        inlier_indices = []
        for i in range(data.shape[1]):
            mx = (parabola[0] * (data[0, i]**2)) + (parabola[1] * data[0, i]) + parabola[2]

            dist[i] = data[1, i] - mx

            # if (i != indices[0] and i != indices[1] and i != indices[2]) and (abs(dist[i]) <= max_noise + 1e-5):
            if (abs(dist[i]) <= max_noise + 1e-5):
                count += 1
                data_inlier.append(data[:, i])
                inlier_indices.append(i)
            
        data_inlier = np.array(data_inlier).T
        
        if count > max_inlier_count:
            max_inlier_count = count
            parabola_fit_all = np.polyfit(data_inlier[0, :], data_inlier[1, :], 2)

        best_guess_history[:, j] = parabola_fit_all

        max_num_inliers_history[j] = max_inlier_count

    return best_guess_history, max_num_inliers_history





