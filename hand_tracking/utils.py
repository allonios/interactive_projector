from enum import Enum
from math import sqrt

import mediapipe as mp

mp_hands = mp.solutions.hands


class Orientation(Enum):
    UNIDENTIFIED = "Unidentified"
    LEFT = "Left"
    RIGHT = "Right"
    UP = "Up"
    DOWN = "Down"

    def __str__(self):
        return self.value


fingers_indexes = {1: 4, 2: 8, 3: 12, 4: 16, 5: 20}


def calculate_distance(point1: int, point2: int, image_shape, landmarks) -> float:
    image_height, image_width, _ = image_shape

    point1_lm = landmarks[point1]
    point2_lm = landmarks[point2]

    return sqrt(
        ((point2_lm.x - point1_lm.x) * image_width) ** 2
        + ((point2_lm.y - point1_lm.y) * image_height) ** 2
    )


def calculate_average_distance(landmarks, image_shape: tuple) -> float:
    sum = 0
    for connection_coords in mp_hands.HAND_CONNECTIONS:
        sum += calculate_distance(
            connection_coords[0], connection_coords[1], image_shape, landmarks
        )
    return sum / 21


def is_above(point1_landmark, point2_landmark) -> bool:
    return point1_landmark.y > point2_landmark.y


def is_down(point1_landmark, point2_landmark) -> bool:
    return point1_landmark.y < point2_landmark.y


def is_right(point1_landmark, point2_landmark) -> bool:
    return point1_landmark.x > point2_landmark.x


def is_left(point1_landmark, point2_landmark) -> bool:
    return point1_landmark.x < point2_landmark.x


class FingerLandmarksPairsFactory:
    @staticmethod
    def get_fingers_landmarks_pairs(hand) -> dict:
        if hand.orientation == Orientation.UP:
            fingers_comparator = is_above
        elif hand.orientation == Orientation.DOWN:
            fingers_comparator = is_down
        elif hand.orientation == Orientation.LEFT:
            fingers_comparator = is_right
        # hand.orientation == Orientation.RIGHT
        else:
            fingers_comparator = is_left

        if hand.thumb_orientation == Orientation.UP:
            thumb_comparator = is_above
        elif hand.thumb_orientation == Orientation.DOWN:
            thumb_comparator = is_down
        elif hand.thumb_orientation == Orientation.LEFT:
            thumb_comparator = is_right
        # hand.thumb_orientation == Orientation.RIGHT
        else:
            thumb_comparator = is_left

        fingers_landmarks_pairs = {
            4: {"threshold": 3, "comparator": thumb_comparator},
            8: {"threshold": 6, "comparator": fingers_comparator},
            12: {"threshold": 10, "comparator": fingers_comparator},
            16: {"threshold": 14, "comparator": fingers_comparator},
            20: {"threshold": 18, "comparator": fingers_comparator},
        }

        return fingers_landmarks_pairs


##########################################################################

fingers_landmarks_pairs = {
    4: {"threshold": 2, "comparator": is_right},
    8: {"threshold": 6, "comparator": is_above},
    12: {"threshold": 10, "comparator": is_above},
    16: {"threshold": 14, "comparator": is_above},
    20: {"threshold": 18, "comparator": is_above},
}


def is_closed(index: int, landmarks):
    point1_index = index
    point2_index = fingers_landmarks_pairs[index]["threshold"]
    comparator = fingers_landmarks_pairs[index]["comparator"]
    return comparator(landmarks[point1_index], landmarks[point2_index])


def check_raised_fingers(landmarks):
    raised = []
    for finger_id, finger_landmark in fingers_indexes.items():
        if not is_closed(finger_landmark, landmarks):
            raised.append(finger_id)

    return raised
