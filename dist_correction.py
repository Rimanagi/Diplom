import cv2
import discorpy

# Определите размеры шахматной доски (количество внутренних углов)
chessboard_size = (8, 8)  # Например, 9x6 внутренних углов шахматной доски

# Создайте списки для хранения изображений шахматной доски и соответствующих углов
chessboard_images = []
chessboard_corners = []

# Открытие видеопотока с видео, содержащим изображения шахматной доски с известными дисторсиями
cap = cv2.VideoCapture(0)  # Замените 'path_to_calibration_video.avi' на путь к вашему видео

while True:
    # Захват кадра с видео
    ret, frame = cap.read(0)

    # Попытка найти углы шахматной доски в текущем кадре
    # ret, corners = cv2.findChessboardCorners(frame, chessboard_size, None)

    # Если углы найдены, добавьте изображение и углы в соответствующие списки
    # if ret:
    #     chessboard_images.append(frame)
    #     chessboard_corners.append(corners)

    # Отображение текущего кадра
    cv2.imshow('Chessboard Calibration', frame)

    # Ожидание нажатия клавиши 'q' для завершения калибровки
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Закрыть видеопоток
cap.release()
cv2.destroyAllWindows()

# Выполнить калибровку камеры на основе изображений шахматной доски и углов
ret, camera_matrix, distortion_coefficients, _, _ = cv2.calibrateCamera(
    chessboard_corners, chessboard_images, frame.shape[1::-1], None, None
)

# Открыть новый видеопоток с видео, которое вы хотите исправить
cap = cv2.VideoCapture('path_to_distorted_video.avi')  # Замените 'path_to_distorted_video.avi' на путь к вашему видео

while True:
    # Захват кадра с видео
    ret, frame = cap.read()

    # Использование функции undistort для коррекции дисторсии в текущем кадре
    undistorted_frame = cv2.undistort(frame, camera_matrix, distortion_coefficients)

    # Отображение оригинального и исправленного кадра
    cv2.imshow('Original Frame', frame)
    cv2.imshow('Undistorted Frame', undistorted_frame)

    # Ожидание нажатия клавиши 'q' для выхода из цикла
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Закрыть видеопоток и окна
cap.release()
cv2.destroyAllWindows()
