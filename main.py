from argparse import ArgumentParser

from image_handlers.image_handler import MediaPipeHandsImageHandler
from image_handlers.stereo_vision_handler import StereoImageHandler
from image_processors.depth_processor import StereoDepthProcessor
from image_processors.hands_centers_processor import HandsCentersProcessor
from image_processors.hands_processor import HandsProcessor
from image_processors.zoom_processor import ZoomProcessor
from projector_detection.functions import calibrate_from_chess_baord


def stereo_setup():
    right_coord_calculator, right_zoom_util = calibrate_from_chess_baord(
        2, 1280, 720
    )
    left_coord_calculator, left_zoom_util = calibrate_from_chess_baord(
        6, 1280, 720
    )
    min_detection_confidence = 0.6

    right_handler_processors = [
        ZoomProcessor(right_zoom_util),
        HandsProcessor(
            min_detection_confidence=min_detection_confidence,
            window_title="right",
        ),
        HandsCentersProcessor(),
    ]

    left_handler_processors = [
        ZoomProcessor(left_zoom_util),
        HandsProcessor(
            min_detection_confidence=min_detection_confidence,
            window_title="left",
        ),
        HandsCentersProcessor(),
    ]

    stereo_processors = [
        StereoDepthProcessor(),
        # StereoHandLocationProcessor(),
        # ClickEventProcessor(coord_calculator),
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
        right_handler, left_handler, baseline=12, processors=stereo_processors
    )

    stereo_vision.run()


def mono_setup():
    # coord_calculator, zoom_util = calibrate_from_chess_baord(
    #     2, 1280, 720
    # )
    min_detection_confidence = 0.6
    handler_processors = [
        # ZoomProcessor(zoom_util),
        HandsProcessor(
            min_detection_confidence=min_detection_confidence,
            window_title="right",
        ),
        HandsCentersProcessor(),
    ]

    handler = MediaPipeHandsImageHandler(
        input_stream=0,
        window_title="right",
        processors=handler_processors,
    )

    handler.run()


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "mode", nargs="?", help="set it to either stereo mode or mono mode"
    )

    args = parser.parse_args()
    if args.mode == "stereo":
        stereo_setup()
    elif args.mode == "mono":
        mono_setup()
    else:
        print("please pick either stereo or mono mode")


if __name__ == "__main__":
    main()
