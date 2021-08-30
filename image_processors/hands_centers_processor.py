from image_processors.base import BaseImageProcessor


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
