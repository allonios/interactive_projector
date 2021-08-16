from multiprocessing import Process, Queue
from queue import Empty
from time import time

import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands


class CameraProcess:
    def __init__(self, source=0, window_title="cam process", max_buffer_size=1):
        self.process = Process()
        self.source = source
        self.cap = cv2.VideoCapture(source)
        self.window_title = window_title
        self.buffer = Queue(maxsize=max_buffer_size)

    def process_input(self):
        while self.cap.isOpened():
            success, image = self.cap.read()

            if not success:
                break

            # cv2.imshow(self.window_title, image)

            # if cv2.waitKey(1) & 0xFF == 27:
            #     break

            yield image

    def read_input(self):
        for image in self.process_input():
            if not self.buffer.full():
                self.buffer.put(image)

    def next_image(self):
        try:
            image = self.buffer.get()
        except Empty:
            image = None

        return image

    def start(self):
        self.process = Process(target=self.read_input)
        self.process.start()

    def stop(self):
        self.cap.release()
        self.process.terminate()


# Testing
# cam1 = CameraProcess()
# cam2 = CameraProcess(2, "2")
#
# cam1.start()
# cam2.start()
#
# while cam1.process.is_alive() and cam1.process.is_alive():
#     cv2.imshow(cam1.window_title, cam1.next_image())
#     cv2.imshow(cam2.window_title, cam2.next_image())
#
#     if cv2.waitKey(1) & 0xFF == 27:
#         cam1.stop()
#         cam2.stop()
#         cv2.destroyAllWindows()


class MediaPipeCameraProcess(CameraProcess):
    def __init__(self, source=0, window_title="cam process", max_buffer_size=1):
        super().__init__(source, window_title, max_buffer_size)
        self.prev_frame_time = 0

    def process_input(self):
        with mp_hands.Hands(
            min_detection_confidence=0.5, min_tracking_confidence=0.5
        ) as hands:
            while self.cap.isOpened():
                success, image = self.cap.read()

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
                results1 = hands.process(image)

                # Draw the hand annotations on the image.
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                if results1.multi_hand_landmarks:
                    for hand_landmarks in results1.multi_hand_landmarks:
                        hand_x, hand_y = (
                            hand_landmarks.landmark[9].x * image.shape[1],
                            hand_landmarks.landmark[9].y * image.shape[0],
                        )

                        cv2.circle(
                            image,
                            (int(hand_x), int(hand_y)),
                            5,
                            (0, 255, 0),
                            cv2.FILLED,
                        )

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

                yield image


# Testing
# cam1 = MediaPipeCameraProcess(0, "cam-process-1")
# cam2 = MediaPipeCameraProcess(2, "cam-process-2")
#
# cam1.start()
# cam2.start()
#
# while cam1.process.is_alive() and cam1.process.is_alive():
#     cv2.imshow(cam1.window_title, cam1.next_image())
#     cv2.imshow(cam2.window_title, cam2.next_image())
#
#     if cv2.waitKey(1) & 0xFF == 27:
#         cam1.stop()
#         cam2.stop()
#         # cv2.destroyAllWindows()
#         exit()
