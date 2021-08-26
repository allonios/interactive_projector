import cv2

cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)


while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("failed to read image")

    cv2.imshow("Image", image)

    image = cv2.resize(image, None, fx=5, fy=5, interpolation=cv2.INTER_LINEAR)

    # cropping the image
    zoomed = image[
        int(image.shape[0] / 5) : int(image.shape[0] / 5) * 2,
        int(image.shape[1] / 5) : int(image.shape[1] / 5) * 2,
    ]

    print(image.shape)

    cv2.imshow("Zoomed", zoomed)

    if cv2.waitKey(1) & 0xFF == 27:
        exit()
