import cv2

from events.events_manager import bus
from image_processors.base import BaseMultipleImagesProcessor

THRESHOLD = 30


class ClickEventProcessor(BaseMultipleImagesProcessor):
    def process_data(self) -> dict:

        depths = self.data["data"]["hands_depths"]

        for index, depth in enumerate(depths):
            hand_id = list(depth.keys())[0]
            hand_depth = list(depth.values())[0]
            if hand_depth <= THRESHOLD:
                event_data = {}
                event_data["right_data"] = self.data["data"]["right_data"]
                event_data["left_data"] = self.data["data"]["left_data"]
                event_data["hand_id"] = hand_id
                event_data["depth"] = hand_depth
                bus.emit("clicked", event_data)

                right_image = self.images[0]
                left_image = self.images[1]

                cv2.putText(
                    right_image,
                    f"clicked: {str(hand_id)}: {str(hand_depth)}",
                    (0, 100 + 30 * index),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (100, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

                cv2.putText(
                    left_image,
                    f"clicked: {str(hand_id)}: {str(hand_depth)}",
                    (0, 100 + 30 * index),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (100, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

        return self.data
