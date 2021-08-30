# from random import randrange

from image_processors.base import BaseImageProcessor


class ZoomProcessor(BaseImageProcessor):
    def __init__(self, callback):
        super().__init__()
        self.callback = callback

    def process_data(self) -> dict:
        zoomed_image = self.callback(self.image, 10, 10, 10, 10)
        self.data["image"] = zoomed_image
        return self.data
