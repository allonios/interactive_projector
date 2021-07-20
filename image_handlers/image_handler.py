from multiprocessing import Manager
from time import time

import cv2
import mediapipe as mp

from image_handlers.base import BaseImageHandlerProcess
from image_processors.hands_centers_processor import HandsCentersProcessor
from image_processors.hands_processor import HandsProcessor

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands


class MediaPipeHandsImageHandler(BaseImageHandlerProcess):
    def __init__(
            self,
            input_stream=0,
            window_title="cam process",
            max_buffer_size=1,
            processors=()
    ):
        super().__init__(input_stream, window_title, max_buffer_size, processors)

        self.prev_frame_time = 0
        self.new_frame_time = 0

        self.current_state = {}

    def display_fps(self):
        font = cv2.FONT_HERSHEY_SIMPLEX
        self.new_frame_time = time()
        fps = 1 / (self.new_frame_time - self.prev_frame_time)
        self.prev_frame_time = self.new_frame_time

        cv2.putText(
            self.current_state["image"],
            "FPS: {:.2f}".format(fps),
            (0, 70),
            font,
            1,
            (100, 255, 0),
            3,
            cv2.LINE_AA,
        )


    def save_info(self, image, info):
        import os, re
        try:
            last_image_count = max(
                map(
                    lambda file_name: int(
                        re.search("(\d+)", file_name).group()),
                    filter(
                        lambda x: x if re.search(
                            "saved-image-\d+", x
                        ) else None,
                        os.listdir("../saved")
                    )
                )
            )
        # ValueError: max() arg is an empty sequence
        except ValueError:
            last_image_count = 0

        cv2.imwrite(
            f"saved/saved-image-{last_image_count + 1}.png",
            image
        )
        file = open(
            f"saved/saved-image-info-{last_image_count + 1}.txt",
            "w"
        )
        file.write(info)
        file.close()


    def handle(self):
        while self.cap.isOpened():
            self.current_state["success"], self.current_state["image"] = self.cap.read()
            self.current_state["data"] = {}
            if not self.current_state["success"]:
                print("Ignoring empty camera frame.")
                continue

            # Flip the image horizontally for a later selfie-view display, and convert
            # the BGR image to RGB.
            self.current_state["image"] = cv2.flip(self.current_state["image"], 1)

            self.implement_processors()


            self.display_fps()

            yield self.current_state

# Testing
# handler1 = MediaPipeHandsImageHandler(0, "1", min_detection_confidence=0.7)
# handler2 = MediaPipeHandsImageHandler(2, "2", min_detection_confidence=0.7)
#
# handler1.start()
# handler2.start()
#
# while handler1.process.is_alive() and handler1.process.is_alive():
#     image1 = handler1.read_next_data()
#     image2 = handler2.read_next_data()
#
#     cv2.imshow(handler1.window_title, image1)
#     cv2.imshow(handler2.window_title, image2)
#
#     if cv2.waitKey(1) & 0xFF == 27:
#         handler1.stop()
#         handler2.stop()
#         # cv2.destroyAllWindows()
#         exit()
