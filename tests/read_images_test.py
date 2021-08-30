import cv2

cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)


while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("failed to read image")

    cv2.imshow("Image", image)

    if cv2.waitKey(1) & 0xFF == 27:
        exit()
