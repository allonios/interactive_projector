# import cv2
#
# cap = cv2.VideoCapture(2)
#
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
#
#
# while cap.isOpened():
#     success, image = cap.read()
#     if not success:
#         print("failed to read image")
#
#     cv2.imshow("Image", image)
#
#     if cv2.waitKey(1) & 0xFF == 27:
#         exit()
import time

import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose

cap = cv2.VideoCapture(0)
prev_frame_time = time.time()
with mp_pose.Pose(
    min_detection_confidence=0.4, min_tracking_confidence=0.3
) as pose:
    while True:
        success, image = cap.read()
        # img_arr = numpy.array(
        #     bytearray(urllib.request.urlopen(URL).read()), dtype=numpy.uint8)
        # image = cv2.imdecode(img_arr, -1)
        image_height, image_width, _ = image.shape

        # Flip the image horizontally for a later selfie-view display, and convert
        # the BGR image to RGB.
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        s = time.time()

        results = pose.process(image)

        print("mp processing time:", time.time() - s)
        # Draw the pose annotation on the image.
        image.flags.writeable = True

        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            # landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
        )
        font = cv2.FONT_HERSHEY_SIMPLEX
        new_frame_time = time.time()
        fps = 1 / (new_frame_time - prev_frame_time)
        prev_frame_time = new_frame_time

        cv2.putText(
            image,
            "FPS: {:.2f}".format(fps),
            (0, 70),
            font,
            1,
            (100, 255, 0),
            3,
            cv2.LINE_AA,
        )
        cv2.imshow("MediaPipe Pose", image)
        if cv2.waitKey(5) & 0xFF == 27:
            break
