import cv2
import numpy as np

# Capturing camera
cap = cv2.VideoCapture(0)  # 0 means using the first free cam we have

while True:
    # Reading frames from camera | ret - return True due to everything is okay
    ret, frame = cap.read()

    # Checking for correctness
    if not ret:
        break

    # Grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Binarization
    _, binary_image = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)

    # Showing the windows
    cv2.imshow('Binary Video', binary_image)
    cv2.imshow('Default Video', frame)
    # cv2.imshow('', gray)

    if cv2.waitKey(1) == 27:  # 27 - code of button 'escape' | (1) запрос каждую миллисекунду
        break

# Closing the windows and turning off camera
cap.release()
cv2.destroyAllWindows()