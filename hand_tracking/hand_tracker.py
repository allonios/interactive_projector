from math import atan2, degrees
from time import time

import cv2
import mediapipe as mp
from utils import (
    FingerLandmarksPairsFactory,
    Orientation,
    calculate_average_distance,
    fingers_indexes,
)

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands


class Hand:
    def __init__(self, hand_landmarks, hand_type, image):
        self.landmarks = hand_landmarks
        self.hand_type = hand_type
        self.image = image
        self.thumb_orientation = Orientation.UNIDENTIFIED
        self.orientation = Orientation.UNIDENTIFIED
        self.__identify_hand_orientation()
        self.__identify_thumb_orientation()

    def __identify_hand_orientation(self):
        middle_finger_tip = self.landmarks.landmark[9]
        hand_center = self.landmarks.landmark[0]

        image_height, image_width, _ = self.image.shape

        # Finger coordinates to the image top left corner (default).
        finger_x = middle_finger_tip.x * image_width
        finger_y = middle_finger_tip.y * image_height

        # New coordinates center.
        center_x = hand_center.x * image_width
        center_y = hand_center.y * image_height

        # Finger coordinates to the image center.
        finger_x = center_x - finger_x
        finger_y = center_y - finger_y

        degree = degrees(atan2(finger_y, finger_x))

        if -135 < degree < -45:
            self.orientation = Orientation.DOWN
        elif 45 < degree < 135:
            self.orientation = Orientation.UP
        elif -45 < degree < 45:
            self.orientation = Orientation.LEFT
        else:
            self.orientation = Orientation.RIGHT

    def __identify_thumb_orientation(self):
        wrist = self.landmarks.landmark[0]
        thumb = self.landmarks.landmark[4]
        if self.orientation == Orientation.UP or self.orientation == Orientation.DOWN:
            if thumb.x < wrist.x:
                self.thumb_orientation = Orientation.LEFT
            else:
                self.thumb_orientation = Orientation.RIGHT

        if (
            self.orientation == Orientation.LEFT
            or self.orientation == Orientation.RIGHT
        ):
            if thumb.y > wrist.y:
                self.thumb_orientation = Orientation.DOWN
            else:
                self.thumb_orientation = Orientation.UP

    def draw_landmarks(self):
        mp_drawing.draw_landmarks(self.image, self.landmarks, mp_hands.HAND_CONNECTIONS)

    def get_depth(self):
        return calculate_average_distance(self.landmarks.landmark, self.image.shape)

    def is_closed_finger(self, index: int):
        fingers_landmarks_pairs = (
            FingerLandmarksPairsFactory.get_fingers_landmarks_pairs(self)
        )
        point1_index = index
        point2_index = fingers_landmarks_pairs[index]["threshold"]
        comparator = fingers_landmarks_pairs[index]["comparator"]
        return comparator(
            self.landmarks.landmark[point1_index], self.landmarks.landmark[point2_index]
        )

    def get_raised_fingers(self):
        raised = []
        for finger_id, finger_landmark in fingers_indexes.items():
            if not self.is_closed_finger(finger_landmark):
                raised.append(finger_id)

        return raised


class HandTracker:
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

        self.detected_hands = []

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

    def track_hands(self):
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

            results = self.hands.process(image)

            # Draw the hand annotations on the image.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            if results.multi_hand_landmarks:
                for hand_landmarks, hand_type in zip(
                    results.multi_hand_landmarks, results.multi_handedness
                ):
                    hand = Hand(hand_landmarks, hand_type, image)
                    hand.draw_landmarks()
                    self.detected_hands.append(hand)

                for hand_index, hand in enumerate(self.detected_hands):
                    print(f"hand id: {hand_index}, distance: {hand.get_depth()}")
                    print(f"hand orientation: {hand.orientation}")
                    print(f"thumb orientation: {hand.thumb_orientation}")
                    print(f"open set: {hand.get_raised_fingers()}")
                    print("_________________________________________________")

                # clear detected hands for a new scan, keep this as
                # the last statement in each iteration.
                self.detected_hands.clear()

            self.display_fps(image)

            cv2.imshow("MediaPipe Hands", image)
            if cv2.waitKey(5) & 0xFF == 27:
                break
