from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

import scipy.io

PATH_TO_DATA = Path(__file__).parent

ALPHA = 0.1
MAX_ITER = 200
BREAK_TERM = 1e-6

# N = 100, K = 21
# q is the positions of the LiDAR detectors.

# p is the actual locations of the tunnel markers.
# p_hat is the noisy measurement
# p_~ is the current estimate of p

# d is the distance between p_hat and q


def main():
    # read .mat files as <numpy.ndarray>'s
    pts_o = scipy.io.loadmat(f"{PATH_TO_DATA}/observation_R5_L40_N100_K21")[
        "pts_o"
    ]  # 21 x 3 q matrix. Contains the positions of each of the 21 sensors.

    pts_markers = scipy.io.loadmat(f"{PATH_TO_DATA}/pts_R5_L40_N100_K21")[
        "pts_markers"
    ]  # 21 x 100 x 3 p_hat matrix.
    # Contains noisy measurements of the tunnel marker positions by each sensor

    dist = scipy.io.loadmat(f"{PATH_TO_DATA}/dist_R5_L40_N100_K21.mat")[
        "dist"
    ]  # 100 x 21 d matrix
    # Contains distance measurements between q and p_hat

    pts_marks_gt = scipy.io.loadmat(f"{PATH_TO_DATA}/gt_R5_L40_N100_K21")[
        "pts_marks_gt"
    ]  # 100 x 3 ground truth tunnel marker position matrix. Used for error calculation

    ##### Part 1: Newton Optimization Algorithm Implementation #####

    # lambda_val = 0.0160603015
    # lambda_val = 0.1
    # p_tilda, num_iters = newton_optimization(
    #     pts_o, pts_markers, dist, lambda_val, 1, MAX_ITER
    # )
    # error = RMSE(p_tilda, pts_marks_gt)
    # print(f"RMSE = {error: .10f} for Lambda = {lambda_val} in {num_iters} iterations")

    ##### Part 2: Fine Tuning Lambda #####

    # optimal_LAMBDA(pts_o, pts_markers, dist, pts_marks_gt)

    ##### Part 3: Fine Tuning Initialization #####

    # optimal_initialization(pts_o, pts_markers, dist, pts_marks_gt)

    return 0


def newton_optimization(
    sensor_positions,
    marker_positions,
    distance_readings,
    lambda_val,
    initialization,
    max_iterations,
):
    # pts_o, pts_markers, dist

    N_num_markers = marker_positions.shape[1]  # 100
    K_num_sensors = sensor_positions.shape[0]  # 21

    p_tilda = np.zeros(shape=(N_num_markers, 3))  # optimized sensor positions matrix

    for i in range(N_num_markers):
        # for the ith marker
        p_hat_i = marker_positions[:, i, :]
        d_i = distance_readings[i, :]

        # initial marker position
        if initialization == 1:
            p = np.mean(
                p_hat_i, axis=0
            )  # initial marker guess is the mean of measured coords
        elif initialization == 2:
            p = p_hat_i[0, :]  # initial marker guess is the first measured coordinates

        elif initialization == 3:
            p = np.random.rand(1, 3)  # initial marker guess is a random vector

        num_iter = 0

        # Perform optimization
        while num_iter < max_iterations:
            residuals = np.linalg.norm(p - sensor_positions, axis=1) - d_i

            Jacobian = (p - sensor_positions) / np.linalg.norm(
                p - sensor_positions, axis=1
            )[:, None]

            gradient = Jacobian.T.dot(residuals) + lambda_val * np.sum(
                2 * (p - p_hat_i), axis=0
            )
            Hessian = Jacobian.T.dot(
                Jacobian
            ) + 2 * lambda_val * K_num_sensors * np.eye(3)

            delta_p = -np.linalg.inv(Hessian).dot(gradient)

            # p for next iteration
            p = p + ALPHA * delta_p

            # Check convergence
            p_tilda[i, :] = p
            num_iter += 1

            if np.linalg.norm(delta_p) < BREAK_TERM:
                break

    return p_tilda, num_iter


def RMSE(p_tilda, p):
    return np.sqrt(np.sum((p_tilda - p) ** 2) / p.size)


def optimal_LAMBDA(sensor_positions, marker_positions, distance_readings, gnd_truth):
    # Create array of lambda values
    lambda_values = np.linspace(1e-3, 1, 200)

    # Initialize array to store RMSE values
    rmse_values = np.zeros_like(lambda_values)

    # Iterate over lambda values
    for j, lambda_val in enumerate(lambda_values):
        # Optimize points and calculate RMSE
        p_tilda, _ = newton_optimization(
            sensor_positions,
            marker_positions,
            distance_readings,
            lambda_val,
            1,
            MAX_ITER,
        )
        rmse_values[j] = RMSE(p_tilda, gnd_truth)

    # Identify lambda value with minimum RMSE
    lambda_min = lambda_values[np.argmin(rmse_values)]

    # Print minimum lambda value
    print(f"Minimum lambda, lambda_op = {lambda_min:.10f}")

    # Plot RMSE values against lambda values
    plt.plot(lambda_values, rmse_values)
    plt.title("RMSE vs Lambda")
    plt.xlabel("Lambda")
    plt.ylabel("RMSE")
    plt.show()


def optimal_initialization(pts_o, pts_markers, dist, pts_markers_gt):
    # plotting RMSE vs number of iterations for each of the 3 initializations
    lambda_val = 0.0160603015  # optimized from step 2

    # 1. Initialization = Average of K coordinates
    # num_ters_1 = []
    # rmse_1 = []

    # for i in range(1, MAX_ITER):
    #     p_tilda, num_iters = newton_optimization(
    #         pts_o, pts_markers, dist, lambda_val, 1, i
    #     )
    #     error = RMSE(p_tilda, pts_markers_gt)

    #     num_ters_1.append(i)
    #     rmse_1.append(error)

    # plt.plot(num_ters_1, rmse_1)
    # plt.title("RMSE vs Number of Iterations with Average Initialization")
    # plt.xlim([1, MAX_ITER])
    # plt.xlabel("Number of Iterations")
    # plt.ylabel("RMSE")
    # print(f"Mimimum RMSE reached in {num_iters} iterations")
    # plt.show()

    # 2. Initialization = First Measured Coordinate
    # num_ters_2 = []
    # rmse_2 = []

    # for i in range(1, MAX_ITER):
    #     p_tilda, num_iters = newton_optimization(
    #         pts_o, pts_markers, dist, lambda_val, 2, i
    #     )
    #     error = RMSE(p_tilda, pts_markers_gt)

    #     num_ters_2.append(i)
    #     rmse_2.append(error)

    # plt.plot(num_ters_2, rmse_2)
    # plt.title("RMSE vs Number of Iterations with Single-Point Initialization")
    # plt.xlim([1, MAX_ITER])
    # plt.xlabel("Number of Iterations")
    # plt.ylabel("RMSE")
    # print(f"Mimimum RMSE reached in {num_iters} iterations")
    # plt.show()

    # 3. Initialization = Random Vector
    num_ters_3 = []
    rmse_3 = []

    for i in range(1, MAX_ITER):
        p_tilda, num_iters = newton_optimization(
            pts_o, pts_markers, dist, lambda_val, 3, i
        )
        error = RMSE(p_tilda, pts_markers_gt)

        num_ters_3.append(i)
        rmse_3.append(error)
    plt.plot(num_ters_3, rmse_3)
    plt.title("RMSE vs Number of Iterations with Random Initialization")
    plt.xlim([1, MAX_ITER])
    plt.xlabel("Number of Iterations")
    plt.ylabel("RMSE")
    print(f"Mimimum RMSE reached in {num_iters} iterations")
    plt.show()


if __name__ == "__main__":
    main()
