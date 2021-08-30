from math import sqrt

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
    def process_data(self) -> dict:
        image = self.image
        hands_data = self.data["data"]["hands_data"]

        for hand_id, hand_data in enumerate(hands_data):
            if not hand_data.get(hand_id).get("in_projector"):
                continue

            wrist = hand_data[hand_id]["wrist"]
            elbow = hand_data[hand_id]["wrist"]

            distance = sqrt(
                (wrist[0] - elbow[0]) ** 2 + (wrist[1] - elbow[1]) ** 2
            )

            distance = max(distance, 100)
            # distance = 100

            min_x = max(0, wrist[0] - distance)
            max_x = min(image.shape[1], wrist[0] + distance)

            min_y = max(0, wrist[1] - distance)
            max_y = min(image.shape[0], wrist[1] + distance)

            cropped_img = image[
                int(min_y) : int(max_y), int(min_x) : int(max_x), :
            ]

            hand_data[hand_id]["hand_image"] = cropped_img
            hand_data[hand_id]["reshape_data"] = {
                "min_x": min_x,
                "max_x": max_x,
                "min_y": min_y,
                "max_y": max_y,
            }

        return self.data
