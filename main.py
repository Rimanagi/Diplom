import cv2
import numpy as np

# Открытие видео с камеры ноутбука (optional)
cap = cv2.VideoCapture(1)

# Коэффициенты искажения для подушкообразной и бочковидной дисторсий
k1, k2, p1, p2 = -0, -0, 0, 0   # k1, k2 - radial dist
                                    # p1, p2 - tangential dist
# k1, k2, p1, p2 = 0, 0, 0, 0
# -0.75
# 0.5, 0.13, 0, 0

while True:
    # Захват кадра с камеры
    ret, frame = cap.read(1)

    # Получение высоты и ширины кадра
    height, width = frame.shape[:2]

    # Генерация матрицы камеры
    camera_matrix = np.array([[width, 0, width / 2],
                              [0, height, height / 2],
                              [0, 0, 1]], dtype=np.float64)

    # Генерация коэффициентов искажения (подушкообразная и бочковидная дисторсии)
    distortion_coefficients = np.array([k1, k2, p1, p2, 0], dtype=np.float64)

    # Применение искажения к кадру
    distorted_frame = cv2.undistort(frame, camera_matrix, distortion_coefficients)

    # Отображение оригинального кадра и кадра с дисторсиями
    cv2.imshow('Original Frame', frame)
    cv2.imshow('Distorted Frame', distorted_frame)

    # Ожидание нажатия клавиши 'q' для выхода из цикла
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Освобождение ресурсов и закрытие окон
cap.release()
cv2.destroyAllWindows()
