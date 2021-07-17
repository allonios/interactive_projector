from multiprocessing import Process

from image_handlers.image_handler import MediaPipeHandsImageHandler
from image_handlers.stereo_vision_handler import StereoImageHandler
from image_processors.hands_centers_processor import HandsCentersProcessor
from image_processors.hands_processor import HandsProcessor

if __name__ == "__main__":
    # handler.handle()

    min_detection_confidence = 0.5

    right_handler_processors = [
        HandsProcessor(
            min_detection_confidence=min_detection_confidence,
            window_title="right",
        ),
        HandsCentersProcessor()
    ]

    left_handler_processors = [
        HandsProcessor(
            min_detection_confidence=min_detection_confidence,
            window_title="left",
        ),
        HandsCentersProcessor()
    ]

    right_handler = MediaPipeHandsImageHandler(
        2,
        "right",
        processors=right_handler_processors,
    )

    left_handler = MediaPipeHandsImageHandler(
        6,
        "left",
        processors=left_handler_processors,
    )

    stereo_vision = StereoImageHandler(right_handler, left_handler, baseline=11)

    stereo_vision.handle()
