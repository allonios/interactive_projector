import cv2

from image_handlers.base import BaseImageHandler


class StereoImageHandler(BaseImageHandler):
    def __init__(
        self,
        right_handler,
        left_handler,
        baseline,
        focal=8,
        alpha=60,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.right_handler = right_handler
        self.left_handler = left_handler
        self.baseline = baseline
        self.focal = focal
        self.alpha = alpha

    def build_new_state(
        self,
        right_image,
        left_image,
        right_hand_image,
        left_hand_image,
        right_image_success,
        left_image_success,
        right_data,
        left_data,
    ):
        self.current_state = {}
        self.current_state["images"] = []
        self.current_state["success"] = []
        self.current_state["data"] = {}

        self.current_state["images"] = {
            "right_image": right_image,
            "left_image": left_image,
            "right_hand_image": right_hand_image,
            "left_hand_image": left_hand_image,
        }

        self.current_state["success"] = {
            "right_image_success": right_image_success,
            "left_image_success": left_image_success,
        }

        self.current_state["data"]["right_data"] = right_data["data"]
        self.current_state["data"]["left_data"] = left_data["data"]

        self.current_state["data"]["baseline"] = self.baseline
        self.current_state["data"]["focal"] = self.focal
        self.current_state["data"]["alpha"] = self.alpha

    def handle(self):
        # self.right_handler.start()
        # self.left_handler.start()

        # while (
        #     self.right_handler.process.is_alive()
        #     and self.left_handler.process.is_alive()
        # ):
        while (
            self.right_handler.cap.isOpened()
            and self.left_handler.cap.isOpened()
        ):
            right_data = self.right_handler.read_next_data()
            left_data = self.left_handler.read_next_data()

            right_image_success = right_data["success"]
            left_image_success = left_data["success"]

            if not right_image_success or not left_image_success:
                continue

            right_image = right_data["image"]
            left_image = left_data["image"]

            # zoomed_right = right_data["zoomed_image"]
            # zoomed_left = left_data["zoomed_image"]

            right_hand_image = right_data.get("hand_image", 0)
            left_hand_image = right_data.get("hand_image", 0)

            # building current state:
            self.build_new_state(
                right_image,
                left_image,
                right_hand_image,
                left_hand_image,
                right_image_success,
                left_image_success,
                right_data,
                left_data,
            )

            # data examples:
            """"""

            self.implement_processors()

            yield self.current_state

    def run(self):
        for data in self.handle():
            left_image = data["images"]["left_image"]
            right_image = data["images"]["right_image"]

            # right_hand_image = data["images"]["right_hand_image"]
            # left_hand_image = data["images"]["left_hand_image"]

            cv2.imshow(self.left_handler.window_title, left_image)
            cv2.imshow(self.right_handler.window_title, right_image)

            right_image_hands_data = data["data"]["right_data"]["hands_data"]
            left_image_hands_data = data["data"]["left_data"]["hands_data"]

            for hand_id, hand_data in enumerate(
                zip(right_image_hands_data, left_image_hands_data)
            ):
                right_hand_data = hand_data[0]
                left_hand_data = hand_data[1]
                if not (
                    right_hand_data.get(hand_id).get("in_projector")
                    and left_hand_data.get(hand_id).get("in_projector")
                ):
                    continue

                right_hand = right_hand_data[hand_id]["hand_image"]
                left_hand = right_hand_data[hand_id]["hand_image"]

                cv2.imshow(
                    self.right_handler.window_title + " zoomed", right_hand
                )
                cv2.imshow(
                    self.left_handler.window_title + " zoomed", left_hand
                )

            # cv2.imshow(self.left_handler.window_title + " zoomed", zoomed_left)
            # cv2.imshow(self.right_handler.window_title + " zoomed", zoomed_right)

            if cv2.waitKey(1) & 0xFF == 27:
                # self.left_handler.stop()
                # self.right_handler.stop()
                exit()
