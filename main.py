from image_handlers.image_handler import ImageHandler

URL = "http://192.168.1.103:8080/video"

handler = ImageHandler(
    min_detection_confidence=0.7,
    input_stream=URL
    #input_stream=2
)

if __name__ == "__main__":
    handler.handle()
