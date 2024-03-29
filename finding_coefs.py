import cv2
import numpy as np

# Загрузите изображение шахматной доски
image_path = '/Users/rimanagi/PycharmProjects/Diplom/lense_2/chess/17cm.png'
image = cv2.imread(image_path)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
k1 = k2 = 0

# Определите количество углов на шахматной доске (например, 9x6)
chessboard_size = (17, 11)

# Найдите углы шахматной доски
ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

# Если углы найдены, используйте их для калибровки камеры
if ret:

    # Точки объекта в реальном мире
    objp = np.zeros((np.prod(chessboard_size), 3), dtype=np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)

    # Точки объекта и точки изображения
    objpoints = [objp]
    imgpoints = [corners]

    # Калибровка камеры
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    # Коэффициенты радиальной дисторсии
    k1, k2 = dist[0][0], dist[0][1]
    print("Коэффициенты радиальной дисторсии: k1 = {}, k2 = {}".format(k1, k2))
else:
    print("Углы шахматной доски не найдены. Попробуйте использовать другое изображение.")


cap = cv2.VideoCapture(1)
ret, frame = cap.read()
height, width = frame.shape[:2]

camera_matrix= np.array([[width, 0, width / 2],
                              [0, height, height / 2],
                              [0, 0, 1]], dtype=np.float64)  # Пример матрицы, замените на вашу
dist = np.array([k1, k2, 0, 0, 0])  # Используем только коэффициенты k1 и k2


while True:
    # Чтение кадра
    ret, frame = cap.read()
    if not ret:
        print("Не удалось захватить видео")
        break

    # Коррекция дисторсии
    h, w = frame.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
    undistorted_frame = cv2.undistort(frame, mtx, dist, None, newcameramtx)

    # Отображение кадра
    cv2.imshow('Undistorted Video', undistorted_frame)

    # Завершение цикла по нажатию клавиши 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Освобождение захвата и закрытие окон
cap.release()
cv2.destroyAllWindows()