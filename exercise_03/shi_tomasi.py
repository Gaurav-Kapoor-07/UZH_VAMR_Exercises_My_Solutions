import numpy as np
from scipy import signal


def shi_tomasi(img, patch_size):
    """ Returns the shi-tomasi scores for an image and patch size patch_size
        The returned scores are of the same shape as the input image """

    pass

    # sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])

    # sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    # I_x = signal.convolve2d(in1=sobel_x, in2=img, mode='valid')

    # I_y = signal.convolve2d(in1=sobel_y, in2=img, mode='valid')

    # Reduce computations

    sobel_x_row = np.array([-1, 0, 1])
    sobel_x_column = np.array([1, 2, 1])

    I_x = signal.convolve2d(img, sobel_x_row[None, :], mode="valid")
    I_x = signal.convolve2d(I_x, sobel_x_column[:, None], mode="valid").astype(float)

    I_y = signal.convolve2d(img, sobel_x_row[:, None], mode="valid")
    I_y = signal.convolve2d(I_y, sobel_x_column[None, :], mode="valid").astype(float)

    I_x_sq = I_x ** 2

    I_y_sq = I_y ** 2

    I_x_I_y = I_x * I_y

    # print(I_x_I_y)

    box_filter = np.ones([patch_size, patch_size]) / (patch_size * patch_size)
    # print(box_filter)

    sum_I_x_sq = signal.convolve2d(in1=box_filter, in2=I_x_sq, mode='valid')
    # print(sum_I_x_sq.shape)

    sum_I_y_sq = signal.convolve2d(in1=box_filter, in2=I_y_sq, mode='valid')
    # print(sum_I_y_sq.shape)

    sum_I_x_I_y = signal.convolve2d(in1=box_filter, in2=I_x_I_y, mode='valid')

    lambda_1 = 0.5 * (sum_I_x_sq + sum_I_y_sq + ((sum_I_x_sq - sum_I_y_sq) ** 2 + 4 * sum_I_x_I_y * sum_I_x_I_y) ** 0.5)
    lambda_2 = 0.5 * (sum_I_x_sq + sum_I_y_sq - ((sum_I_x_sq - sum_I_y_sq) ** 2 + 4 * sum_I_x_I_y * sum_I_x_I_y) ** 0.5)

    lambda_2[lambda_2 < 0] = 0

    pr = patch_size // 2
    lambda_2 = np.pad(lambda_2, [(pr+1, pr+1), (pr+1, pr+1)], mode='constant', constant_values=0)

    # print(img.shape)
    # print(lambda_2.shape)

    return lambda_2