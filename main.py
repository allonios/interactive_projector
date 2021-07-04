from image_handler import ImageHandler

URL = "http://192.168.1.101:8080/video"

handler = ImageHandler(
    min_detection_confidence=0.7,
    # input_stream=URL
)

if __name__ == "__main__":
    handler.handle()
