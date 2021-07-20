from image_processors.base import BaseMultipleImagesProcessor
from utils.stereo_vision.triangulation import find_depth
import cv2

class DepthProcessor(BaseMultipleImagesProcessor):
    def process_data(self):
        right_image = self.images[0]
        left_image = self.images[1]

        baseline = self.data["data"]["baseline"]
        focal = self.data["data"]["focal"]
        alpha = self.data["data"]["alpha"]

        right_centers_of_hands = self.data["data"]["right_data"]["hands_centers"]
        left_centers_of_hands = self.data["data"]["left_data"]["hands_centers"]

        self.data["data"]["hands_depths"] = []

        for right_hand_info, left_hand_info in zip(
                right_centers_of_hands,
                left_centers_of_hands
        ):
            hand_id = list(right_hand_info.keys())[0]

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

            print("depth")
            print(depth)

            self.data["data"]["hands_depths"].append(
                {
                    hand_id: depth
                }
            )

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

        return self.data
