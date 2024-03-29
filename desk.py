import cv2
import numpy as np

def find_distortion_coefficients(image_path, chessboard_size=(11, 7)):
    # Загрузка изображения
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Нахождение углов шахматной доски
    ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

    if ret:
        # Критерии остановки (точность/количество итераций)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

        # Подготовка точек в реальном мире
        objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)

        # Калибровка камеры
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera([objp], [corners], gray.shape[::-1], None, None)
        return mtx, dist
    else:
        print("Не удалось найти углы шахматной доски.")
        return None, None

def undistort_live_video(dist_coeffs, camera_index=0):
    cap = cv2.VideoCapture(camera_index)

    # Получаем первоначальные параметры изображения
    ret, frame = cap.read(2)
    if ret:
        h, w = frame.shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(dist_coeffs[0], dist_coeffs[1], (w, h), 1, (w, h))

        # Цикл захвата и коррекции видео
        while True:
            ret, frame = cap.read(2)
            if ret:
                # Коррекция дисторсии с использованием оптимальной новой матрицы камеры
                dst = cv2.undistort(frame, dist_coeffs[0], dist_coeffs[1], None, newcameramtx)

                # Обрезка изображения для удаления черных краев
                x, y, w, h = roi
                dst = dst[y:y+h, x:x+w]

                cv2.imshow('Undistorted Video', dst)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                print("Не удалось получить кадр.")
                break

    cap.release()
    cv2.destroyAllWindows()


image_path = 'Pictures/chess_table_1.jpg'
mtx, dist = find_distortion_coefficients(image_path)

if mtx is not None and dist is not None:
    undistort_live_video((mtx, dist))
else:
    print("Не удалось вычислить коэффициенты дисторсии.")
