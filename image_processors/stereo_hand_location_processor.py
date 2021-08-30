from image_processors.base import BaseMultipleImagesProcessor


class StereoHandLocationProcessor(BaseMultipleImagesProcessor):
    def process_data(self) -> dict:
        right_centers = self.data["data"]["right_data"]["hands_centers"]
        left_centers = self.data["data"]["left_data"]["hands_centers"]

        # "hands_centers": [0: (x1, y1), 1: (x2, y2), ]

        hands_data = self.data["data"]["hands_data"]

        for right_center, left_center in zip(right_centers, left_centers):
            hand_id = list(right_center.keys())[0]
            right_center_x = list(right_center.values())[0][0]
            left_center_x = list(left_center.values())[0][0]

            new_hand_center_x = (left_center_x + right_center_x) / 2
            # the hand has the same Ys in both images
            hand_center_y = list(right_center.values())[0][1]

            hands_data[hand_id]["center"] = (new_hand_center_x, hand_center_y)

        # we might not need this line.
        self.data["data"]["hands_data"] = hands_data

        return self.data
