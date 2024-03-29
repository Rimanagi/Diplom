import cv2
import numpy as np
import glob

# Настройки шахматной доски
chessboard_size = (17, 11)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Подготовка точек в реальном мире
objp = np.zeros((np.prod(chessboard_size), 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)

# Массивы для хранения точек объекта и точек изображения из всех изображений
objpoints = []  # 3d точки в реальном мире
imgpoints = []  # 2d точки на плоскости изображения

# Загрузка изображений для калибровки
images = glob.glob('/Users/rimanagi/PycharmProjects/Diplom/lense_2/chess/10cm.jpg')
                   # '/Users/rimanagi/PycharmProjects/Diplom/lense_2/chess/10cm.jpg',
                   # '/Users/rimanagi/PycharmProjects/Diplom/lense_2/chess/17cm.jpg')

# Инициализация переменных для размера изображения
h, w = 0, 0

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = img.shape[:2]  # Обновляем размеры изображения

    # Находим углы на доске
    ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

    if ret == True:
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners)

        # Отображаем углы
        cv2.drawChessboardCorners(img, chessboard_size, corners2, ret)
        cv2.imshow('img', img)
        cv2.waitKey(500)

cv2.destroyAllWindows()

# Убедитесь, что h и w были получены
if h == 0 or w == 0:
    raise ValueError("Не удалось определить размеры изображения для калибровки.")

# Калибровка камеры
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, (w, h), None, None)

# Получение новой матрицы камеры
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))


# Функция для исправления дисторсии
def undistort_frame(frame, mtx, dist, newcameramtx, roi):
    dst = cv2.undistort(frame, mtx, dist, None, newcameramtx)
    x, y, w, h = roi
    dst = dst[y:y + h, x:x + w]
    return dst


# Захват видео с камеры
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_undistorted = undistort_frame(frame, mtx, dist, newcameramtx, roi)
    cv2.imshow('Undistorted', frame_undistorted)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
