from image_processors.base import BaseImageProcessor


class InProjectorProcessor(BaseImageProcessor):
    def __init__(self, callback):
        super().__init__()
        self.callback = callback

    def process_data(self) -> dict:
        hands_data = self.data["data"]["hands_data"]

        for hand_id, hand_data in enumerate(hands_data):
            wrist = hand_data[hand_id]["wrist"]
            in_projector, _ = self.callback(wrist)
            hand_data[hand_id]["in_projector"] = in_projector

        return self.data
