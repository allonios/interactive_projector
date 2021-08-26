from copy import deepcopy
from enum import Enum
from math import cos, sin, sqrt

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


FINGERS_INDEXES = {1: 4, 2: 8, 3: 12, 4: 16, 5: 20}


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


def calculate_multiple_distances_to_point(
    from_point, to_points, landmarks, image_shape
) -> float:
    sum = 0
    for to_point in to_points:
        if from_point != to_point:
            sum += calculate_distance(from_point, to_point, image_shape, landmarks)
    return sum


def calculate_collection_average_distance(
    landmarks_indexes: list, landmarks, image_shape: tuple
):
    sum = 0
    for landmark_index in landmarks_indexes:
        sum += calculate_multiple_distances_to_point(
            landmark_index, landmarks_indexes, landmarks, image_shape
        )
    return sum / len(landmarks_indexes) * (len(landmarks_indexes) - 1)


def get_rotate_landmarks(landmarks, degree):
    rotated_landmarks = deepcopy(landmarks)
    for landmark in rotated_landmarks:
        rotated_x = landmark.x * cos(degree) + landmark.y * sin(degree)
        rotated_y = landmark.y * cos(degree) - landmark.x * sin(degree)
        landmark.x = rotated_x
        landmark.y = rotated_y
    return rotated_landmarks


class FingerLandmarksPairsFactory:
    @classmethod
    def __get_thumb_orientation_after_rotation(cls, wrist, thumb):
        if wrist.x > thumb.x:
            return Orientation.LEFT
        else:
            return Orientation.RIGHT

    @classmethod
    def get_fingers_landmarks_pairs(cls, hand) -> dict:
        orientation = cls.__get_thumb_orientation_after_rotation(
            hand.rotated_landmarks[0], hand.rotated_landmarks[1]
        )
        if orientation == Orientation.LEFT:
            thumb_comparator = lambda point1, point2: point1.x > point2.x
        # hand.thumb_orientation == Orientation.RIGHT
        else:
            thumb_comparator = lambda point1, point2: point1.x < point2.x

        fingers_landmarks_pairs = {
            4: {"threshold": 3, "comparator": thumb_comparator},
            8: {
                "threshold": 5,
                "comparator": lambda point1, point2: point1.y > point2.y,
            },
            12: {
                "threshold": 9,
                "comparator": lambda point1, point2: point1.y > point2.y,
            },
            16: {
                "threshold": 13,
                "comparator": lambda point1, point2: point1.y > point2.y,
            },
            20: {
                "threshold": 17,
                "comparator": lambda point1, point2: point1.y > point2.y,
            },
        }

        return fingers_landmarks_pairs
