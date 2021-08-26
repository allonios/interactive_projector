from image_handlers.image_handler import MediaPipeHandsImageHandler
from image_handlers.stereo_vision_handler import StereoImageHandler
from image_processors.click_processor import ClickEventProcessor
from image_processors.depth_processor import DepthProcessor
from image_processors.hands_centers_processor import HandsCentersProcessor
from image_processors.hands_processor import HandsProcessor
from image_processors.stereo_hand_location_processor import \
    StereoHandLocationProcessor

if __name__ == "__main__":
    min_detection_confidence = 0.7

    right_handler_processors = [
        HandsProcessor(
            min_detection_confidence=min_detection_confidence,
            window_title="right",
        ),
        HandsCentersProcessor(),
    ]

    left_handler_processors = [
        HandsProcessor(
            min_detection_confidence=min_detection_confidence,
            window_title="left",
        ),
        HandsCentersProcessor(),
    ]

    stereo_processors = [
        DepthProcessor(),
        StereoHandLocationProcessor(),
        ClickEventProcessor(),
    ]

    right_handler = MediaPipeHandsImageHandler(
        input_stream=2,
        window_title="right",
        processors=right_handler_processors,
    )

    left_handler = MediaPipeHandsImageHandler(
        input_stream=6,
        window_title="left",
        processors=left_handler_processors,
    )

    stereo_vision = StereoImageHandler(
        right_handler, left_handler, baseline=22, processors=stereo_processors
    )

    stereo_vision.handle()
