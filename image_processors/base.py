from abc import ABCMeta, abstractmethod
from typing import Tuple

from numpy import ndarray


class BaseImageProcessor(metaclass=ABCMeta):
    def __init__(self):
        self.image = None

    def __call__(self, image: ndarray):
        self.image = image
        return self.process_image()

    @abstractmethod
    def process_image(self) -> Tuple[ndarray, None]:
        """
        return values:
        ndarray: represents the processed image.
        None: represents extra data about the processed image.
        """
        pass
