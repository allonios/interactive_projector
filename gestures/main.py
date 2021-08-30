import enum

import cv2
import mediapipe as mp
import numpy
import urllib.request
from math import pi
mp_drawing = mp.solutions.drawing_utils
# mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

class Direction(enum.Enum):
    TOP = 0
    BOTTOM = 1
    RIGHT = 2
    LEFT = 3
    TOP_LEFT = 4
    BOTTOM_LEFT = 5
    TOP_RIGHT = 6
    BOTTOM_RIGHT = 7

URL = "http://192.168.46.114:8080/shot.jpg"

old_pos = numpy.array([0,0])
state = "None"
size_of_buffer = 10
directions_buffer = []
circle_gesture = [ Direction.TOP, Direction.TOP_LEFT ,Direction.LEFT, Direction.BOTTOM_LEFT, Direction.BOTTOM, Direction.BOTTOM_RIGHT, Direction.RIGHT, Direction.TOP_RIGHT ]
snake_gesture = [ Direction.TOP_RIGHT ,Direction.RIGHT, Direction.BOTTOM_RIGHT, Direction.BOTTOM, Direction.BOTTOM_RIGHT, Direction.RIGHT, Direction.TOP_RIGHT ]

def get_direction(deg):
    if deg == 90:
        return Direction.TOP
    if deg == -90:
        return Direction.BOTTOM
    if deg == 180 or deg == -180:
        return Direction.LEFT
    if deg == 135:
        return Direction.TOP_LEFT
    if deg == -135:
        return Direction.BOTTOM_LEFT
    if deg == 0:
        return Direction.RIGHT
    if deg == 45:
        return Direction.TOP_RIGHT
    if deg == -45:
        return Direction.BOTTOM_RIGHT
    else:
        print("deg => ", deg)
        return 0
def get_direction_text(direction):
    if direction == Direction.TOP:
        return "TOP"
    if direction == Direction.BOTTOM:
        return "BOTTOM"
    if direction == Direction.LEFT:
        return "LEFT"
    if direction == Direction.RIGHT:
        return "RIGHT"
    if direction == Direction.TOP_RIGHT:
        return "TOP_RIGHT"
    if direction == Direction.BOTTOM_RIGHT:
        return "BOTTOM_RIGHT"
    if direction == Direction.TOP_LEFT:
        return "TOP_LEFT"
    if direction == Direction.BOTTOM_LEFT:
        return "BOTTOM_LEFT"


def buffer_equality(buffer, gesture, from_index, to_index):
    for i in range(from_index, to_index):
        if buffer[i] != gesture[i - from_index]:
            return False
    return True


def search_in_buffer(buffer, gesture):
    buffer_length = len(buffer)
    gesture_length = len(gesture)
    for i,_ in enumerate(buffer):
        if(i + gesture_length >= buffer_length):
            break
        if buffer_equality(buffer, gesture, i, i + gesture_length):
            return True
    return False




def add_to_buffer(direction):
    if(len(directions_buffer) > 0 and directions_buffer[-1] == direction):
        print("returning")
        return

    if(len(directions_buffer) < size_of_buffer):
        directions_buffer.append(direction)
    else:
        directions_buffer.pop(0)
        directions_buffer.append(direction)

# cap = cv2.VideoCapture(0)
with mp_pose.Pose(
    min_detection_confidence=0.4,
    min_tracking_confidence=0.3) as pose:
  while True:
    # success, image = cap.read()
    img_arr = numpy.array(
        bytearray(urllib.request.urlopen(URL).read()), dtype=numpy.uint8)
    image = cv2.imdecode(img_arr, -1)
    image_height, image_width, _ = image.shape

    # Flip the image horizontally for a later selfie-view display, and convert
    # the BGR image to RGB.
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    results = pose.process(image)
    # Draw the pose annotation on the image.
    image.flags.writeable = True

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    mp_drawing.draw_landmarks(
        image,
        results.pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        # landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
    )
    if results.pose_landmarks:
        x = results.pose_landmarks.landmark[15].x * image_width
        y = results.pose_landmarks.landmark[15].y * image_height
        visibility = results.pose_landmarks.landmark[15].visibility

        if visibility > 0.5:
            vector = numpy.array([x - old_pos[0], y - old_pos[1]])
            mag = numpy.sqrt(vector.dot(vector))
            if mag > 15 :
                angle = -numpy.angle(complex(vector[0], vector[1]), deg=True)
                deg = round(angle / 45) * 45
                old_pos = numpy.array([x, y])
                cv2.putText(image, str(mag), (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
                direction = get_direction(deg)
                direction_text = get_direction_text(direction)
                add_to_buffer(direction)
                cv2.putText(image, str(direction_text), (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                is_circle = search_in_buffer(directions_buffer, circle_gesture)
                is_snake = search_in_buffer(directions_buffer, snake_gesture)
                if is_circle:
                    directions_buffer = []
                    print("Circle")
                    state = "Circle"
                if is_snake:
                    directions_buffer = []
                    print("SNAKE")
                    state = "Snake"


            cv2.circle(image,(int(x),int(y)), 3, (255, 0, 0), -1)

        cv2.putText(image, str(state), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
    cv2.imshow('MediaPipe Pose', image)
    if cv2.waitKey(5) & 0xFF == 27:
      break
