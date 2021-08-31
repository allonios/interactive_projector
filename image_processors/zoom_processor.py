from math import sqrt

import cv2

from image_processors.base import BaseImageProcessor


class ZoomProcessor(BaseImageProcessor):
    def __init__(self, callback):
        super().__init__()
        self.callback = callback

    def process_data(self) -> dict:
        zoomed_image = self.callback(self.image, 10, 10, 10, 10)
        self.data["image"] = zoomed_image
        return self.data


class HandCropperProcessor(BaseImageProcessor):
    def rescale_frame(self, frame, scale=0.50):
        width = int(frame.shape[1] * scale)
        height = int(frame.shape[0] * scale)
        dimensions = (width, height)

        return cv2.resize(frame, dimensions, interpolation=cv2.INTER_AREA)

    def process_data(self) -> dict:
        image = self.image
        hands_data = self.data["data"]["hands_data"]

        for hand_id, hand_data in enumerate(hands_data):
            if not hand_data.get(hand_id).get("in_projector"):
                continue

            wrist = hand_data[hand_id]["wrist"]
            elbow = hand_data[hand_id]["elbow"]

            distance = sqrt(
                (wrist[0] - elbow[0]) ** 2 + (wrist[1] - elbow[1]) ** 2
            )

            distance = distance / 2

            distance = max(distance, 50)

            min_x = max(0, wrist[0] - distance)
            max_x = min(image.shape[1], wrist[0] + distance)

            min_y = max(0, wrist[1] - distance)
            max_y = min(image.shape[0], wrist[1] + distance)

            cropped_img = image[
                int(min_y) : int(max_y), int(min_x) : int(max_x), :
            ]

            scale = 500 / cropped_img.shape[1]

            cropped_img = self.rescale_frame(cropped_img, scale)

            hand_data[hand_id]["hand_image"] = cropped_img
            hand_data[hand_id]["reshape_data"] = {
                "min_x": min_x,
                "max_x": max_x,
                "min_y": min_y,
                "max_y": max_y,
                "scale": scale,
            }

        return self.data
