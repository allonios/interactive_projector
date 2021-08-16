import cv2

from image_handlers.base import BaseImageHandler


class StereoImageHandler(BaseImageHandler):
    def __init__(
        self, right_handler, left_handler, baseline, focal=8, alpha=60, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.right_handler = right_handler
        self.left_handler = left_handler
        self.baseline = baseline
        self.focal = focal
        self.alpha = alpha

    def handle(self):
        self.right_handler.start()
        self.left_handler.start()

        while (
            self.right_handler.process.is_alive()
            and self.left_handler.process.is_alive()
        ):
            right_data = self.right_handler.read_next_data()
            left_data = self.left_handler.read_next_data()

            right_image_success = right_data["success"]
            left_image_success = left_data["success"]

            if not right_image_success or not left_image_success:
                continue

            right_image = right_data["image"]
            left_image = left_data["image"]

            right_centers_of_hands = right_data["data"].get("hands_centers", None)
            left_centers_of_hands = left_data["data"].get("hands_centers", None)

            # print("right data", right_data)

            # building current state:
            self.build_new_state(
                right_image,
                left_image,
                right_image_success,
                left_image_success,
                right_data,
                left_data,
            )

            # right_image, left_image = undistorted_rectify(right_image, left_image)

            # data examples:
            """
            current_state: {
                "images": [image1, image2, ]
                "success": [success_image1, success_image2, ]
                "data": {
                    "baseline": baseline,
                    "focal": focal,
                    "alpha": alpha,
                    "right_data": {
                        "hands_centers": [0: (x1, y1), 1: (x2, y2), ]
                    }
                    "left_data": {
                        "hands_centers": [0: (x1, y1), 1: (x2, y2), ]
                    }
                }
            }
            """

            """
            new_state: {
                "images": [image1, image2, ]
                "success": [success_image1, success_image2, ]
                "data": {
                    "baseline": baseline,
                    "focal": focal,
                    "alpha": alpha,
                    "right_data": {
                        "hands_centers": [0: (x1, y1), 1: (x2, y2), ],
                    }
                    "left_data": {
                        "hands_centers": [0: (x1, y1), 1: (x2, y2), ],
                    }
                    "hands_depths": [0: depth, 1: depth, ]

                }
            }
            """

            if right_centers_of_hands and left_centers_of_hands:
                self.implement_processors()

            cv2.imshow(self.left_handler.window_title, left_image)
            cv2.imshow(self.right_handler.window_title, right_image)

            if cv2.waitKey(1) & 0xFF == 27:
                self.left_handler.stop()
                self.right_handler.stop()
                # cv2.destroyAllWindows()
                exit()

    def build_new_state(
        self,
        right_image,
        left_image,
        right_image_success,
        left_image_success,
        right_data,
        left_data,
    ):
        self.current_state = {}
        self.current_state["images"] = []
        self.current_state["success"] = []
        self.current_state["data"] = {}

        self.current_state["images"].append(right_image)
        self.current_state["images"].append(left_image)

        self.current_state["success"].append(right_image_success)
        self.current_state["success"].append(left_image_success)

        self.current_state["data"]["right_data"] = right_data["data"]
        self.current_state["data"]["left_data"] = left_data["data"]

        self.current_state["data"]["baseline"] = self.baseline
        self.current_state["data"]["focal"] = self.focal
        self.current_state["data"]["alpha"] = self.alpha
