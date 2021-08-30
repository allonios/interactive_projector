import cv2
from numpy import zeros

from events.events_manager import bus
from image_processors.base import BaseMultipleImagesProcessor

THRESHOLD = 45


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
