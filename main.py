from argparse import ArgumentParser

from image_handlers.image_handler import MediaPipeHandsImageHandler
from image_handlers.stereo_vision_handler import StereoImageHandler
from image_processors.getures_processor import GesturesProcessor
from image_processors.hands_centers_processor import \
    PoseBasedHandsCentersProcessor
from image_processors.hands_processor import PoseBasedHandsProcessor


def stereo_setup():
    # right_coord_calculator, right_zoom_util = calibrate_from_chess_baord(
    #     2, 1280, 720
    # )
    # left_coord_calculator, left_zoom_util = calibrate_from_chess_baord(
    #     6, 1280, 720
    # )
    # import cv2
    #
    # cv2.destroyWindow("chessboard")

    pose_min_detection_confidence = 0.4
    hand_min_detection_confidence = 0.4

    right_handler_processors = [
        # ZoomProcessor(right_zoom_util),
        # HandsProcessor(
        #     min_detection_confidence=min_detection_confidence,
        #     window_title="right",
        # ),
        PoseBasedHandsProcessor(
            min_detection_confidence=pose_min_detection_confidence,
            window_title="right",
        ),
        PoseBasedHandsCentersProcessor(),
        # InProjectorProcessor(right_coord_calculator),
        # HandCropperProcessor(),
        # HandsProcessorV2(
        #     min_detection_confidence=hand_min_detection_confidence,
        #     window_title="right",
        # ),
        # HandsCentersProcessorV2(),
    ]

    left_handler_processors = [
        # ZoomProcessor(left_zoom_util),
        PoseBasedHandsProcessor(
            min_detection_confidence=pose_min_detection_confidence,
            window_title="left",
        ),
        PoseBasedHandsCentersProcessor(),
        GesturesProcessor(),
        # InProjectorProcessor(left_coord_calculator),
        # HandCropperProcessor(),
        # HandsProcessorV2(
        #     min_detection_confidence=hand_min_detection_confidence,
        #     window_title="left",
        # ),
        # HandsCentersProcessorV2(),
    ]

    stereo_processors = [
        # StereoDepthProcessor(right_coord_calculator, left_coord_calculator),
        # ClickEventProcessorV2(),
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
    min_detection_confidence = 0.4
    handler_processors = [
        # ZoomProcessor(zoom_util),
        PoseBasedHandsProcessor(
            min_detection_confidence=min_detection_confidence,
            window_title="window",
        ),
        # HandsProcessor(
        #     min_detection_confidence=min_detection_confidence,
        #     window_title="right",
        # ),
        # HandsCentersProcessor(),
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
