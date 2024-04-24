import cv2
import numpy as np
import glob

import find_foef
import plot_creation

# Настройки шахматной доски
chessboard_size = (17, 11)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.00001)

# Подготовка точек в реальном мире
objp = np.zeros((np.prod(chessboard_size), 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)

# Массивы для хранения точек объекта и точек изображения из всех изображений
objpoints = []  # 3d точки в реальном мире
imgpoints = []  # 2d точки на плоскости изображения
temp = '/Users/rimanagi/PycharmProjects/Diplom/lense_None/screenshot 2024-04-17 at 11.48.20 pm.jpg'

# Загрузка изображений для калибровки
paths = [
    # glob.glob('/Users/rimanagi/PycharmProjects/Diplom/lense_None/*.jpg'),
    glob.glob(temp),
    glob.glob('/Users/rimanagi/PycharmProjects/Diplom/lense_2/sequence2/*.jpg'),
    glob.glob('/Users/rimanagi/PycharmProjects/Diplom/lense_2_corrected/*.jpg'),
]

all_dist_coefficients = []

for images in paths:
    # Инициализация переменных для размера изображения
    h, w = 0, 0
    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = img.shape[:2]  # Обновляем размеры изображения

        # Находим углы на доске
        ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

        if ret:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners)

            '''Uncomment to show captured ChessCorners'''
            # cv2.drawChessboardCorners(img, chessboard_size, corners2, ret)
            # cv2.imshow('img', img)
            # cv2.waitKey(20000)

            # if cv2.waitKey(0) & 0xFF == ord('q'):
            #     break

    cv2.destroyAllWindows()

    # Убедитесь, что h и w были получены
    print("w:", w, 'h:', h)
    if h == 0 or w == 0:
        raise ValueError("Не удалось определить размеры изображения для калибровки.")

    # Калибровка камеры
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, (w, h), None, None)
    all_dist_coefficients.append(dist[0])
    # dist[0][2] = dist[0][3] = 0 # Тангенциальная дисторсия

    print('dist coefficients: ', dist, '\n', sep='\n')
    # mtx = np.array([[w, 0, w / 2],
    #                 [0, h, h / 2],
    #                 [0, 0, 1]], dtype=np.float64)

    # np.array([[464.88576345, 0, 994.51364844],
    #           [0, 466.04785799, 543.38488636],
    #           [0., 0., 1.]], dtype=np.float64)
    print('mtx:', mtx, '\n', sep='\n')

    # Получение новой матрицы камеры
    new_camera_mtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
    print('new_mtx:', new_camera_mtx, '\n', sep='\n')

    '''Changing new_camera_mts'''


    # print('mtx=', mtx, sep='\n')
    # new_camera_mtx[0][0] = new_camera_mtx[0][0] * (new_camera_mtx[1][1] / new_camera_mtx[0][0])
    # print('newmtw', new_camera_mtx, sep='\n')
    # print(roi)
    # x, y = 0, 0
    # roi = (x, y, 1920, 1080)

    # Функция для исправления дисторсии
    def undistorsing_frame(frame, mtx, dist, newcameramtx, roi):
        dst = cv2.undistort(frame, mtx, dist, None, newcameramtx)
        x, y, w, h = roi
        dst = dst[y:y + h, x:x + w]
        return dst


    '''un/comment to show corrected pictures'''
    # for fname in images:
    #     frame_undistorted = undistort_frame(cv2.imread(fname), mtx, dist, mtx, roi)
    #     while True:
    #         cv2.imshow('Undistorted', frame_undistorted)
    #         if cv2.waitKey(1) & 0xFF == ord('q'):
    #             break

    # Захват видео с камеры
    cap = cv2.VideoCapture(0)
    zero_koeffs = np.array([0, 0, 0, 0, 0], dtype=np.float64)
    while True:
        ret, frame = cap.read(0)
        if not ret:
            break
        height, width = frame.shape[:2]
        original_frame = cv2.undistort(frame, new_camera_mtx, zero_koeffs)
        cv2.imshow('Original', original_frame)

        frame_undistorted = undistorsing_frame(frame, mtx, dist, new_camera_mtx, roi)
        cv2.imshow('Undistorted', frame_undistorted)

        distorted_frame = cv2.undistort(frame, mtx, dist)
        cv2.imshow('Undistorted', distorted_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # cap.release()
    # cv2.destroyAllWindows()

    # [[624.49723087   0.         915.11659329]
    #  [  0.         271.70313103 709.94978252]
    #  [  0.           0.           1.        ]]

    # np.array([[width, 0, width / 2],
    # [0, height, height / 2],
    # [0, 0, 1]], dtype=np.float64)

find_foef.find_distorsed_points(all_dist_coefficients)

print(*all_dist_coefficients, sep='\n')
plot_creation.plot_creation()

#  в пояснительную записку включить сравнение тангенциальных коэффициентов
#  фокусное внешнего объектива 1.22 mm

# сделать фото с телефона (калибровочные)
# пересчет эквивалентного фокусного расстояния в реальное
# квадрат предметной координаты на графике (r -> r**2)
# 0; 0,5; 0,707 ; 1
# подписать оси по всем точкам
# поянительная записка
# грант
# довавить фотки