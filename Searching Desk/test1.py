import cv2
import numpy as np

# Загрузка изображения
img = cv2.imread('/Pictures/chess_table_1.jpg')

# Определение размеров изображения
height, width = img.shape[:2]


focal_length = 1000
# Расчет координат центра
center_x = width / 2
center_y = height / 2

# Параметры камеры, полученные в результате калибровки
camera_matrix = np.array([[focal_length, 0, center_x],
                          [0, focal_length, center_y],
                          [0, 0, 1]])


k1, k2 = -0.5, -0.13
p1 = p2 = 0
k3 = 0
dist_coeffs = np.array([k1, k2, p1, p2, k3])  # коэффициенты дисторсии

# Коррекция дисторсии
undistorted_img = cv2.undistort(img, camera_matrix, dist_coeffs)

# Отображение результата
cv2.imshow('Undistorted Image', undistorted_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
