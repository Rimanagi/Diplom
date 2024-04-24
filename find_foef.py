# [[-0.47854145  0.21994439  0.00277357 -0.00125357 -0.04258208]]

x = []
y = []


def find_distorsed_points(all_coefficients, image=None, step=0.1):
    for coefficients in all_coefficients:
        temp_x, temp_y = [], []
        k1, k2, k3 = coefficients[0], coefficients[1], coefficients[-1]
        print('radial coefficients: ', k1, k2, k3)

        undistorsed_points = []
        distorsed_points = []

        # setting start points
        height, width = 1080, 1920
        start_point_y = int(height / 2)
        start_point_x = int(width / 2)

        shift_y = step * start_point_y
        shift_x = step * start_point_x

        current_point_y = start_point_y + shift_y
        current_point_x = start_point_x + shift_x

        r = step
        while r <= 1.0:
            curr_x = start_point_x + (r * start_point_x)
            curr_y = start_point_y + (r * start_point_y)
            undistorsed_points.append(curr_y)

            # calculating distorsed coordinates
            x_dist = curr_x * (1 + k1 * (r ** 2) + k2 * (r ** 4) + k3 * (r ** 6))
            y_dist = curr_y * (1 + k1 * (r ** 2) + k2 * (r ** 4) + k3 * (r ** 6))
            distorsed_points.append(y_dist)
            temp_y.append(r)
            current_point_y += shift_y
            r += step

        print("orig: ", undistorsed_points, '\n', 'dist: ', distorsed_points, sep='')

        for i in range(len(distorsed_points)):
            temp_x.append((undistorsed_points[i] - distorsed_points[i]) / undistorsed_points[i])
        x.append(temp_x)
        y.append(temp_y)