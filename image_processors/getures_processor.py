from events.events_manager import bus
from gestures.functions import Gestures
from image_processors.base import BaseImageProcessor


class GesturesProcessor(BaseImageProcessor):
    def __init__(self, data=None):
        super().__init__(data)
        self.gestures_objects = [Gestures(), Gestures()]

    def process_data(self) -> dict:
        hands_data = self.data["data"]["hands_data"]

        for hand_id, hand_data in enumerate(hands_data):
            wrist = hand_data[hand_id]["wrist"]
            _, state = self.gestures_objects[hand_id].update(wrist)
            if state:
                bus.emit(
                    "gesture_raised",
                    {
                        "gesture_name": state,
                    },
                )

        return self.data
