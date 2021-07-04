import cv2
import mediapipe as mp

from time import time
from image_processors.hands_processor import HandsProcessor

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands


class ImageHandler():
    def __init__(
            self,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            max_num_hands=2,
            input_stream=0
    ):
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.max_num_hands = max_num_hands

        self.prev_frame_time = 0
        self.new_frame_time = 0

        self.cap = cv2.VideoCapture(input_stream)

        self.current_image = None

        self.hands_processor = HandsProcessor(
            self.min_detection_confidence,
            self.min_tracking_confidence,
            self.max_num_hands,
        )

    def display_fps(self):
        font = cv2.FONT_HERSHEY_SIMPLEX
        self.new_frame_time = time()
        fps = 1 / (self.new_frame_time - self.prev_frame_time)
        self.prev_frame_time = self.new_frame_time

        cv2.putText(
            self.current_image,
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

            self.current_image, detected_hands = self.hands_processor(
                self.current_image)
            info = ""
            for hand_index, hand in enumerate(detected_hands):
                info = info + \
                       f"hand id: {hand_index}, distance: {hand.get_depth()}\n" \
                       f"hand orientation: {hand.orientation}\n" \
                       f"thumb orientation: {hand.thumb_orientation}\n" \
                       f"open set: {hand.get_raised_fingers()}\n" \
                       "_________________________________________________\n"
            if info:
                print(info)

            self.display_fps()

            cv2.imshow("MediaPipe Hands", self.current_image)
            key = cv2.waitKey(1)

            if key & 0xFF == 27:
                break
            elif key == ord("s"):
                import os, re
                try:
                    last_image_count = max(
                        map(
                            lambda file_name: int(re.search("(\d+)", file_name).group()),
                            filter(
                                lambda x: x if re.search(
                                    "saved-image-\d+", x
                                ) else None,
                                os.listdir("saved")
                            )
                        )
                    )
                # ValueError: max() arg is an empty sequence
                except ValueError:
                    last_image_count = 0

                cv2.imwrite(
                    f"saved/saved-image-{last_image_count+1}.png",
                    self.current_image
                )
                file = open(
                    f"saved/saved-image-info-{last_image_count+1}.txt",
                    "w"
                )
                file.write(info)
                file.close()
