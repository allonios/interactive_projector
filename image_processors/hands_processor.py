from math import atan2, degrees, radians
from typing import List, Tuple

import cv2
import mediapipe as mp
from numpy import ndarray

from image_processors.base import BaseImageProcessor
from utils.utils import (
    FINGERS_INDEXES,
    FingerLandmarksPairsFactory,
    Orientation,
    get_rotate_landmarks, calculate_collection_average_distance,
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
        degree = self.get_hand_rotation_degrees()

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
        thumb = self.landmarks.landmark[2]
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
        return calculate_collection_average_distance(
            [0, 1, 2, 5, 9, 13, 17],
            self.landmarks.landmark,
            self.image.shape
        )
        # return calculate_average_distance(self.landmarks.landmark, self.image.shape)

    def get_hand_rotation_degrees(self):
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

        return degrees(atan2(finger_y, finger_x))

    def is_closed_finger(self, landmark_index: int):
        fingers_landmarks_pairs = (
            FingerLandmarksPairsFactory.get_fingers_landmarks_pairs(self)
        )
        point1_index = landmark_index
        point2_index = fingers_landmarks_pairs[landmark_index]["threshold"]
        comparator = fingers_landmarks_pairs[landmark_index]["comparator"]
        return comparator(
            self.rotated_landmarks[point1_index], self.rotated_landmarks[point2_index]
        )

    def get_raised_fingers(self):
        raised = []
        rotation = self.get_hand_rotation_degrees()
        rotated_landmarks = get_rotate_landmarks(
            self.landmarks.landmark, radians(rotation - 90)
        )
        self.rotated_landmarks = rotated_landmarks
        for finger_id, finger_landmark in FINGERS_INDEXES.items():
            if not self.is_closed_finger(finger_landmark):
                raised.append(finger_id)

        return raised


class HandsProcessor(BaseImageProcessor):
    def __init__(
            self,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            max_num_hands=2
    ):
        super().__init__()
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.max_num_hands = max_num_hands

        self.hands = mp_hands.Hands(
            max_num_hands=self.max_num_hands,
            min_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence,
        )

    def process_image(self) -> Tuple[ndarray, List[Hand]]:
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)

        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        self.image.flags.writeable = False

        results = self.hands.process(self.image)

        # Draw the hand annotations on the image.
        self.image.flags.writeable = True
        self.image = cv2.cvtColor(self.image, cv2.COLOR_RGB2BGR)

        detected_hands = []
        if results.multi_hand_landmarks:
            for hand_landmarks, hand_type in zip(
                results.multi_hand_landmarks, results.multi_handedness
            ):
                hand = Hand(hand_landmarks, hand_type, self.image)
                hand.draw_landmarks()
                detected_hands.append(hand)

        return self.image, detected_hands
