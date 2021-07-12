import cv2

cap = cv2.VideoCapture(2)

while cap.isOpened():
    success, image = cap.read()

    cv2.imshow("Test", image)

    if cv2.waitKey(1) & 0xFF == 27:
        exit()
