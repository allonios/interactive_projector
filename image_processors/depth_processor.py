from image_processors.base import (BaseImageProcessor,
                                   BaseMultipleImagesProcessor)
from utils.stereo_vision.triangulation import find_depth


class StereoDepthProcessor(BaseMultipleImagesProcessor):
    def __init__(self, callback_right, callback_left):
        super().__init__()
        self.callback_right = callback_right
        self.callback_left = callback_left

    def original_coord_calculator(self, point, padding_x, padding_y, scale):
        result = (
            int(padding_x + (point[0] / scale)),
            int(padding_y + (point[1] / scale)),
        )
        # print(point, padding_x, padding_y, scale, result)
        return result

    def build_hand_state(self, hand_id):
        self.data["data"]["hands_data"][hand_id] = {}
        self.data["data"]["hands_data"][hand_id]["depth"] = 0

    def process_data(self):
        right_image_hands_data = self.data["data"]["right_data"]["hands_data"]
        left_image_hands_data = self.data["data"]["left_data"]["hands_data"]

        self.data["data"]["merged_data"] = {}

        if not (right_image_hands_data and left_image_hands_data):
            return self.data

        right_image = self.images["right_image"]
        left_image = self.images["left_image"]

        baseline = self.data["data"]["baseline"]
        alpha = self.data["data"]["alpha"]

        for hand_id, hand_data in enumerate(
            zip(right_image_hands_data, left_image_hands_data)
        ):
            right_image_hand_data = hand_data[0].get(hand_id)
            left_image_hand_data = hand_data[1].get(hand_id)

            right_cropped_image_hand_coord = right_image_hand_data.get(
                "detected_hand_coords"
            )
            left_cropped_image_hand_coord = left_image_hand_data.get(
                "detected_hand_coords"
            )

            if not (
                right_cropped_image_hand_coord
                and left_cropped_image_hand_coord
            ):
                continue

            right_image_hand_coord = self.original_coord_calculator(
                right_cropped_image_hand_coord,
                right_image_hand_data["reshape_data"]["min_x"],
                right_image_hand_data["reshape_data"]["min_y"],
                right_image_hand_data["reshape_data"]["scale"],
            )

            left_image_hand_coord = self.original_coord_calculator(
                left_cropped_image_hand_coord,
                left_image_hand_data["reshape_data"]["min_x"],
                left_image_hand_data["reshape_data"]["min_y"],
                left_image_hand_data["reshape_data"]["scale"],
            )

            # cv2.line(
            #     right_image,
            #     (
            #         int(right_image_hand_data["reshape_data"]["min_x"]),
            #         50,
            #     ),
            #     (
            #         int(right_image_hand_data["reshape_data"]["min_x"]),
            #         right_image.shape[0],
            #     ),
            #     (0, 0, 255),
            #     3
            # )
            #
            # cv2.line(
            #     right_image,
            #     (
            #         50,
            #         int(right_image_hand_data["reshape_data"]["min_y"]),
            #     ),
            #     (
            #         right_image.shape[1],
            #         int(right_image_hand_data["reshape_data"]["min_y"]),
            #     ),
            #     (0, 0, 255),
            #     3
            # )
            #
            # cv2.circle(
            #     right_image,
            #     (
            #         int(right_image_hand_coord[0]),
            #         int(right_image_hand_coord[1]),
            #     ),
            #     5,
            #     (255, 255, 0),
            #     -1
            # )

            depth = find_depth(
                right_image_hand_coord,
                left_image_hand_coord,
                right_image,
                left_image,
                baseline,
                alpha,
            )

            print("depth", hand_id, depth)

            _, right_image_hand_coord = self.callback_right(
                right_image_hand_coord
            )
            _, left_image_hand_coord = self.callback_left(
                left_image_hand_coord
            )

            self.data["data"]["merged_data"][hand_id] = {
                "depth": depth,
                "hand_coord": (
                    (right_image_hand_coord[0] + left_image_hand_coord[0]) / 2,
                    right_image_hand_coord[1],
                ),
            }

        return self.data


class MonocularDepthProcessor(BaseImageProcessor):
    def process_data(self) -> dict:
        # image = self.image
        pass
