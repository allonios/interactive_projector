from abc import ABCMeta, abstractmethod


class BaseImageProcessor(metaclass=ABCMeta):
    """
    class for base processors
    should accept a dictionary representing the current state of the processed image in the pipe
    and should return a new dictionary representing the new state of the processed image in the pipe.
    """

    def __init__(self, data=None):
        self.data = data
        if not isinstance(data, dict):
            self.data = {}
        self.image = self.data.get("image", None)
        self.success = self.data.get("success", None)

    def __call__(self, data: dict):
        self.data = data
        self.image = data["image"]
        return self.process_data()

    @abstractmethod
    def process_data(self) -> dict:
        """
        return values:
        a dict with this main key:value "image": ndarray representing the processed image.
        a dict with this key:value "success": bool.
        data: a dict with the extra data values.
        example:
        {
            "image": self.image,
            "success": self.success,
            "data": {
                "some_data1": value1,
                "some_data2": value2,
            }
        }
        """
        pass

    def __str__(self):
        return ""
