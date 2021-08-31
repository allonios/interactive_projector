import cv2
from numpy import zeros

from events.events_manager import bus
from image_processors.base import BaseMultipleImagesProcessor

THRESHOLD = 400


class ClickEventProcessor(BaseMultipleImagesProcessor):
    def __init__(self, callback):
        super().__init__()
        self.callback = callback

    def process_data(self) -> dict:
        hands_data = self.data["data"]["hands_data"]
        for hand_id in hands_data:
            hand_depth = hands_data[hand_id]["depth"]
            if hand_depth <= THRESHOLD:
                hand_center = hands_data[hand_id]["center"]
                in_projector, in_screen_click = self.callback(hand_center)
                if in_projector:
                    event_data = {
                        "hand_id": hand_id,
                        "hand_depth": hand_depth,
                        "hand_center": hand_center,
                        "in_screen_click": in_screen_click,
                    }
                    bus.emit("clicked", event_data)
                    black = zeros((480, 640, 3))
                    cv2.circle(
                        black,
                        (int(in_screen_click[0]), int(in_screen_click[1])),
                        5,
                        (0, 255, 0),
                        cv2.FILLED,
                    )
                    cv2.imshow("Location", black)
                    cv2.waitKey(1)
                else:
                    cv2.destroyWindow("Location")
            else:
                cv2.destroyWindow("Location")

        return self.data


class ClickEventProcessorV2(BaseMultipleImagesProcessor):
    def __init__(self):
        super().__init__()
        self.buffer = []

    def process_data(self) -> dict:
        right_image_hands_data = self.data["data"]["right_data"]["hands_data"]
        left_image_hands_data = self.data["data"]["left_data"]["hands_data"]

        merged_data = self.data["data"]["merged_data"]

        if not (right_image_hands_data and left_image_hands_data):
            return self.data

        for hand_id, hand_data in enumerate(
            zip(right_image_hands_data, left_image_hands_data)
        ):
            right_image_hand_data = hand_data[0].get(hand_id)
            left_image_hand_data = hand_data[1].get(hand_id)

            if (
                right_image_hand_data["in_projector"]
                and left_image_hand_data["in_projector"]
                and merged_data
            ):
                hand_depth = merged_data[hand_id]["depth"]
                # if hand_depth >= THRESHOLD:
                if True:
                    # self.buffer.append(hand_depth)
                    hand_coords = merged_data[hand_id]["hand_coord"]
                    bus.emit(
                        "clicked",
                        {
                            "hand_id": hand_id,
                            "hand_depth": hand_depth,
                            "hand_coords": hand_coords,
                        },
                    )
                # else:
                #     self.buffer.clear()
                #
                # if len(self.buffer) > 1:
                #     hand_coords = merged_data[hand_id]["hand_coord"]
                #     bus.emit("clicked", {
                #         "hand_id": hand_id,
                #         "hand_depth": hand_depth,
                #         "hand_coords": hand_coords,
                #     })

        return self.data
