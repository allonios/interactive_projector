from time import time

import cv2
import mediapipe as mp

from image_processors.hands_processor import HandsProcessor

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands


class ImageHandler:
    def __init__(
        self,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        max_num_hands=2,
    ):
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.max_num_hands = max_num_hands

        self.prev_frame_time = 0
        self.new_frame_time = 0

        self.cap = cv2.VideoCapture(0)

        self.hands = mp_hands.Hands(
            max_num_hands=self.max_num_hands,
            min_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence,
        )
        self.current_image = None
        self.hands_processor = HandsProcessor(
            self.min_detection_confidence,
            self.min_tracking_confidence,
            self.max_num_hands,
        )

    def display_fps(self, image):
        font = cv2.FONT_HERSHEY_SIMPLEX
        self.new_frame_time = time()
        fps = 1 / (self.new_frame_time - self.prev_frame_time)
        self.prev_frame_time = self.new_frame_time

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

    def handle(self):
        while self.cap.isOpened():
            success, self.current_image = self.cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                # If loading a video, use 'break' instead of 'continue'.
                continue

            # Flip the image horizontally for a later selfie-view display, and convert
            # the BGR image to RGB.
            self.current_image = cv2.cvtColor(
                cv2.flip(self.current_image, 1), cv2.COLOR_BGR2RGB
            )

            self.current_image, detected_hands = self.hands_processor(self.current_image)

            for hand_index, hand in enumerate(detected_hands):
                print(f"hand id: {hand_index}, distance: {hand.get_depth()}")
                print(f"hand orientation: {hand.orientation}")
                print(f"thumb orientation: {hand.thumb_orientation}")
                print(f"open set: {hand.get_raised_fingers()}")
                print("_________________________________________________")

            self.display_fps(self.current_image)

            cv2.imshow("MediaPipe Hands", self.current_image)
            if cv2.waitKey(5) & 0xFF == 27:
                break
