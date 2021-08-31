from image_processors.base import BaseImageProcessor


class HandsCentersProcessorV2(BaseImageProcessor):
    def process_data(self) -> dict:
        hands_data = self.data["data"]["hands_data"]

        for hand_id, hand_data in enumerate(hands_data):
            if not hand_data.get(hand_id).get("in_projector"):
                continue

            detected_hand = hand_data[hand_id]["detected_hand"]
            if detected_hand:
                hand_image = hand_data[hand_id]["hand_image"]
                hand_data[hand_id]["detected_hand_coords"] = (
                    detected_hand.landmarks.landmark[0].x
                    * hand_image.shape[1],
                    detected_hand.landmarks.landmark[0].y
                    * hand_image.shape[0],
                )
        return self.data


class HandsCentersProcessor(BaseImageProcessor):
    def process_data(self) -> dict:
        self.data["data"]["hands_centers"] = list(
            map(
                lambda hand: {
                    hand.id: (
                        hand.landmarks.landmark[9].x * self.image.shape[1],
                        hand.landmarks.landmark[9].y * self.image.shape[0],
                    )
                },
                self.data["data"]["detected_hands"],
            )
        )
        # removing detected hands object because it contains some google mp objects
        # that can't be serialized for buffering with multiprocessing.
        del self.data["data"]["detected_hands"]
        return self.data


class PoseBasedHandsCentersProcessor(BaseImageProcessor):
    def process_data(self) -> dict:
        self.data["data"]["hands_data"] = list(
            map(
                lambda hand: {
                    hand.id: {
                        "wrist": hand.wrist,
                        "elbow": hand.elbow,
                    }
                },
                self.data["data"]["detected_hands"],
            )
        )
        # removing detected hands object because it contains some google mp objects
        # that can't be serialized for buffering with multiprocessing.
        del self.data["data"]["detected_hands"]
        return self.data
