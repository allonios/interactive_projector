import enum
import time

import numpy
import random
import cv2
import mediapipe as mp
import urllib.request
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
import json

URL = "http://192.168.46.114:8080/shot.jpg"

class Gestures():
    def __init__(self, size_of_buffer = 10, config_file_url = "gestures.json", minimum_magnitude= 15):
        self.size_of_buffer = size_of_buffer
        self.old_pos = numpy.array([0,0])
        self.state = state = "None"
        self.directions_buffer = []
        self.config_file_url = config_file_url
        self.load_config_file()
        self.minimum_magnitude = minimum_magnitude


    def buffer_equality(self, buffer, gesture, from_index, to_index):
        for i in range(from_index, to_index):
            if buffer[i] != gesture[i - from_index]:
                return False
        return True

    def search_in_buffer(self, gesture):
        buffer_length = len(self.directions_buffer)
        gesture_length = len(gesture)
        for i, _ in enumerate(self.directions_buffer):
            if (i + gesture_length >= buffer_length):
                break
            if self.buffer_equality(self.directions_buffer, gesture, i, i + gesture_length):
                return True
        return False

    def add_to_buffer(self, direction):
        if direction == -1:
            return

        if (len(self.directions_buffer) > 0 and self.directions_buffer[-1] == direction):
            return

        if (len(self.directions_buffer) < self.size_of_buffer):
            self.directions_buffer.append(direction)
        else:
            self.directions_buffer.pop(0)
            self.directions_buffer.append(direction)

    def get_direction(self, point):
        vector = numpy.array([point[0] - self.old_pos[0], point[1] - self.old_pos[1]])
        mag = numpy.sqrt(vector.dot(vector))
        if mag > self.minimum_magnitude:
            angle = -numpy.angle(complex(vector[0], vector[1]), deg=True)
            deg = round(angle / 45) * 45
            if deg == -180:
                deg = 180
            self.old_pos = point
            return deg, mag
        return -1, -1

    def check_gestures(self):
        for gesture in self.gestures:
            if self.search_in_buffer(gesture["directions"]):
                self.directions_buffer = []
                eval(gesture["callback"])
                return gesture["state"]


    def update(self, point):
            direction, _ = self.get_direction(point)
            if direction == -1:
                return direction, self.state
            self.add_to_buffer(direction)
            self.state = self.check_gestures()
            return direction, self.state

    def add_gesture(self, gesture, add_to_config = True):
        self.gestures.append(gesture)
        if add_to_config:
            f = open(self.config_file_url, "w")
            f.write(json.dumps(self.gestures))


    def load_config_file(self):
        f = open(self.config_file_url, "r")
        json_string = f.read()
        json_object = json.loads(json_string)
        self.gestures = json_object


def get_direction_text(direction):
    if direction == 90:
        return "TOP"
    if direction == -90:
        return "BOTTOM"
    if direction == 180:
        return "LEFT"
    if direction == 0:
        return "RIGHT"
    if direction == 45:
        return "TOP_RIGHT"
    if direction == -45:
        return "BOTTOM_RIGHT"
    if direction == 135:
        return "TOP_LEFT"
    if direction == -135:
        return "BOTTOM_LEFT"

def apply_gestures(gestures):
    LANDMARK = 15

    state = "None"

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

        # mp_drawing.draw_landmarks(
        #     image,
        #     results.pose_landmarks,
        #     mp_pose.POSE_CONNECTIONS,
        #     # landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
        # )

        if results.pose_landmarks:
            x = results.pose_landmarks.landmark[LANDMARK].x * image_width
            y = results.pose_landmarks.landmark[LANDMARK].y * image_height
            visibility = results.pose_landmarks.landmark[LANDMARK].visibility
            if visibility > 0.5:
                direction, tstate = gestures.update((x, y))
                if tstate != None:
                    state = tstate
                cv2.circle(image,(int(x),int(y)), 3, (255, 0, 0), -1)
                cv2.putText(image, get_direction_text(direction), (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)

            cv2.putText(image, str(state), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
            cv2.imshow('MediaPipe Pose', image)
            if cv2.waitKey(5) & 0xFF == 27:
              break

def normalize(buffer, mag_buffer):
    if len(buffer) == 0:
        return buffer
    new_buffer = []

    np_mag_buffer = numpy.array(mag_buffer)
    avg_mag = numpy.mean(np_mag_buffer)
    avg_mag = avg_mag * 0.5

    for index, direction in enumerate(buffer):
        if mag_buffer[index] >= avg_mag:
            new_buffer.append(direction)

    return new_buffer


def add_gesture(gestures):
    LANDMARK = 15
    buffer = []
    mag_buffer = []
    record = False
    start_time = time.time()
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
                x = results.pose_landmarks.landmark[LANDMARK].x * image_width
                y = results.pose_landmarks.landmark[LANDMARK].y * image_height
                visibility = results.pose_landmarks.landmark[LANDMARK].visibility
                if visibility > 0.5:
                    point = (x,y)
                    direction, mag = gestures.get_direction(point)
                    print(direction)

                    if not record:
                        if direction != -1:
                            start_time = time.time()
                        else:
                            if time.time() - start_time > 5:
                                record = True
                                start_time = time.time()
                    else:
                        if direction != -1:
                            start_time = time.time()
                            if len(buffer) > 0 and direction == buffer[-1]:
                                continue
                            buffer.append(direction)
                            mag_buffer.append(mag)
                        else:
                            if time.time() - start_time > 5:
                                break

                else:
                    start_time = time.time()

                cv2.putText(image, "Record" if record else "Waiting", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                (255, 0, 0), 2)
                cv2.putText(image, str(time.time() - start_time), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                cv2.putText(image, str(normalize(buffer, mag_buffer)), (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                cv2.circle(image, (int(x), int(y)), 3, (255, 0, 0), -1)
                cv2.imshow('MediaPipe Pose', image)
                if cv2.waitKey(5) & 0xFF == 27:
                    break

    buffer = normalize(buffer, mag_buffer)
    gesture = {
        "state": f"GESTURES{random.random()}",
        "directions": buffer,
        "callback": "print('custom')"
    }
    print(gesture)
    print(buffer)
    gestures.add_gesture(gesture)

gestures = Gestures(minimum_magnitude=40)

add_gesture(gestures)
gestures.load_config_file()
apply_gestures(gestures)
