from math import atan2, degrees
from time import time

import cv2
import mediapipe as mp
from utils import calculate_average_distance, check_raised_fingers

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

prev_frame_time = 0
new_frame_time = 0

cap = cv2.VideoCapture(0)
with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            continue

        # Flip the image horizontally for a later selfie-view display, and convert
        # the BGR image to RGB.
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        results = hands.process(image)

        # Draw the hand annotations on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                print(
                    "avg distance",
                    calculate_average_distance(hand_landmarks.landmark, image.shape),
                )

                print(check_raised_fingers(hand_landmarks.landmark))
                for index, landmark in enumerate(hand_landmarks.landmark):
                    h, w, c = image.shape
                    cx, cy = landmark.x * w, landmark.y * h

                    # print(f"landmark {index} in position x: {cx} y: {cy}")
                    if index == 8:
                        h, w, c = image.shape
                        cx, cy = landmark.x * w, landmark.y * h

                        image_center_x = w / 2
                        image_center_y = h / 2

                        acx = image_center_x - cx
                        acy = image_center_y - cy

                        degree = degrees(atan2(acy, acx))

                        text = "None"

                        if -135 < degree < -45:
                            print("Bottom")
                        elif 45 < degree < 135:
                            print("Up")
                        elif -45 < degree < 45:
                            print("Left")
                        else:
                            print("Right")

                        print("degree: ", degree)

                        cv2.circle(
                            image,
                            (int(image_center_x), int(image_center_y)),
                            5,
                            (0, 255, 0),
                            cv2.FILLED,
                        )

                    # #     cv2.circle (
                    # #     image,
                    # #     (int(cx),
                    # #     int(cy)),
                    # #     25,
                    # #     (0, 255, 0),
                    # #     cv2.FILLED
                    # # )
                    #     print("landmark Zs: {:.3}".format(landmark.z))

                mp_drawing.draw_landmarks(
                    image, hand_landmarks, mp_hands.HAND_CONNECTIONS
                )

        font = cv2.FONT_HERSHEY_SIMPLEX
        new_frame_time = time()
        fps = 1 / (new_frame_time - prev_frame_time)
        prev_frame_time = new_frame_time
        cv2.putText(
            image,
            "FPS: {:.2f}".format(fps),
            (7, 70),
            font,
            3,
            (100, 255, 0),
            3,
            cv2.LINE_AA,
        )

        cv2.imshow("MediaPipe Hands", image)
        if cv2.waitKey(5) & 0xFF == 27:
            break
cap.release()
