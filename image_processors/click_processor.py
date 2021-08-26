import cv2
from numpy import zeros

from events.events_manager import bus
from image_processors.base import BaseMultipleImagesProcessor

THRESHOLD = 45


class ClickEventProcessor(BaseMultipleImagesProcessor):
    def process_data(self) -> dict:
        hands_data = self.data["data"]["hands_data"]

        for hand_id in hands_data:
            hand_depth = hands_data[hand_id]["depth"]
            if hand_depth <= THRESHOLD:
                hand_center = hands_data[hand_id]["center"]
                event_data = {
                    "hand_id": hand_id,
                    "hand_depth": hand_depth,
                    "hand_center": hand_center,
                }
                bus.emit("clicked", event_data)
                black = zeros((480, 640, 3))
                cv2.circle(
                    black,
                    (int(hand_center[0]), int(hand_center[1])),
                    5,
                    (0, 255, 0),
                    cv2.FILLED,
                )
                cv2.imshow("Location", black)
                cv2.waitKey(1)
            else:
                cv2.destroyWindow("Location")

        return self.data
