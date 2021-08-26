import cv2

from image_processors.base import BaseMultipleImagesProcessor
from utils.stereo_vision.triangulation import find_depth


class DepthProcessor(BaseMultipleImagesProcessor):
    def build_hand_state(self, hand_id):
        self.data["data"]["hands_data"][hand_id] = {}
        self.data["data"]["hands_data"][hand_id]["depth"] = 0

    def process_data(self):
        right_image = self.images["right_image"]
        left_image = self.images["left_image"]

        baseline = self.data["data"]["baseline"]
        focal = self.data["data"]["focal"]
        alpha = self.data["data"]["alpha"]

        right_centers_of_hands = self.data["data"]["right_data"][
            "hands_centers"
        ]
        left_centers_of_hands = self.data["data"]["left_data"]["hands_centers"]

        self.data["data"]["hands_data"] = {}

        for right_hand_info, left_hand_info in zip(
            right_centers_of_hands, left_centers_of_hands
        ):
            hand_id = list(right_hand_info.keys())[0]

            self.build_hand_state(hand_id)

            right_hand_center = list(right_hand_info.values())[0]
            left_hand_center = list(left_hand_info.values())[0]

            depth = find_depth(
                right_hand_center,
                left_hand_center,
                right_image,
                left_image,
                baseline,
                focal,
                alpha,
            )

            # print("depth:", depth)

            self.data["data"]["hands_data"][hand_id]["depth"] = depth

            cv2.putText(
                right_image,
                f"depth: {str(round(depth, 1))}",
                (0, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.5,
                (100, 255, 0),
                3,
                cv2.LINE_AA,
            )

            cv2.putText(
                left_image,
                f"depth: {str(round(depth, 1))}",
                (0, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.5,
                (100, 255, 0),
                3,
                cv2.LINE_AA,
            )

        # print("hands data in depth processor:")
        # print(self.data["data"]["hands_data"])

        return self.data
