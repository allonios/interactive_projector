from turtle import right

import cv2

from utils.stereo_vision.calibration import undistortRectify
from utils.stereo_vision.triangulation import find_depth


class StereoImageHandler:
    def __init__(self, right_handler, left_handler, baseline, focal=8, alpha=60):
        self.right_handler = right_handler
        self.left_handler = left_handler
        self.baseline = baseline
        self.focal = focal
        self.alpha = alpha

    def handle(self):
        self.right_handler.start()
        self.left_handler.start()

        while self.right_handler.process.is_alive() and self.left_handler.process.is_alive():
            right_data = self.right_handler.read_next_data()
            left_data = self.left_handler.read_next_data()

            right_image = right_data["image"]
            left_image = left_data["image"]

            right_image_success = right_data["success"]
            left_image_success = left_data["success"]

            right_centers_of_hands = right_data["data"].get("hands_centers", None)
            left_centers_of_hands = left_data["data"].get("hands_centers", None)

            if not right_image_success or not left_image_success:
                continue

            # right_image, left_image = undistortRectify(right_image, left_image)


            if right_centers_of_hands and left_centers_of_hands:
                for right_hand_center, left_hand_center in zip(right_centers_of_hands, left_centers_of_hands):

                    depth = find_depth(
                        right_hand_center,
                        left_hand_center,
                        right_image,
                        left_image,
                        self.baseline,
                        self.focal,
                        self.alpha,
                    )

                    print("depth")
                    print(depth)

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


            cv2.imshow(self.left_handler.window_title, left_image)
            cv2.imshow(self.right_handler.window_title, right_image)

            if cv2.waitKey(1) & 0xFF == 27:
                self.left_handler.stop()
                self.right_handler.stop()
                # cv2.destroyAllWindows()
                exit()
