from image_handlers.image_handler import ImageHandler, \
    MediaPipeHandsImageHandler
from image_handlers.stereo_vision_handler import StereoImageHandler

URL = "http://192.168.1.103:8080/video"

# handler = ImageHandler(
#     min_detection_confidence=0.7,
#     input_stream=URL
#     #input_stream=2
# )

if __name__ == "__main__":
    # handler.handle()

    right_handler = MediaPipeHandsImageHandler(
        2,
        "2",
        min_detection_confidence=0.7
    )

    left_handler = MediaPipeHandsImageHandler(
        0,
        "1",
        min_detection_confidence=0.7
    )

    stereo_vision = StereoImageHandler(right_handler, left_handler, 19)

    stereo_vision.handle()
