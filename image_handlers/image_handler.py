from time import time

import cv2
import mediapipe as mp

from image_handlers.base import BaseImageHandlerProcess
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
                                os.listdir("../saved")
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


class MediaPipeHandsImageHandler(BaseImageHandlerProcess):
    def __init__(
            self,
            input_stream=0,
            window_title="cam process",
            max_buffer_size=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            max_num_hands=2,

    ):
        super().__init__(input_stream, window_title, max_buffer_size)

        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.max_num_hands = max_num_hands

        self.prev_frame_time = 0
        self.new_frame_time = 0

        self.current_image = None

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
            success, self.current_image = self.cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                # If loading a video, use 'break' instead of 'continue'.
                continue

            # Flip the image horizontally for a later selfie-view display, and convert
            # the BGR image to RGB.
            self.current_image = cv2.flip(self.current_image, 1)

            # implement your processors here
            self.hands_processor = HandsProcessor(
                self.min_detection_confidence,
                self.min_tracking_confidence,
                self.max_num_hands,
            )
            self.current_image, detected_hands = self.hands_processor(
                self.current_image)


            info = ""
            for hand_index, hand in enumerate(detected_hands):
                info = info + \
                       f"window: {self.window_title}\n" \
                       f"hand id: {hand_index}, distance: {hand.get_depth()}\n" \
                       f"hand orientation: {hand.orientation}\n" \
                       f"thumb orientation: {hand.thumb_orientation}\n" \
                       f"open set: {hand.get_raised_fingers()}\n" \
                       "____________________________________________________\n"
            # if info:
            #     print(info)

            self.display_fps()

            centers_of_hands = list(
                map(
                    lambda hand: (
                        hand.landmarks.landmark[9].x * self.current_image.shape[1],
                        hand.landmarks.landmark[9].y * self.current_image.shape[0],
                    ),
                    detected_hands
                )
            )

            return_data = {
                "image": self.current_image,
                "success": success,
                "centers_of_hands": centers_of_hands
            }

            yield return_data

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
