import cv2

cap = cv2.VideoCapture(6)

while cap.isOpened():
    success, image = cap.read()

    if not success:
        print("failed to read image")

    cv2.imshow("Test", image)

    if cv2.waitKey(1) & 0xFF == 27:
        exit()
