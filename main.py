from image_handler import ImageHandler

handler = ImageHandler(min_detection_confidence=0.9)

if __name__ == "__main__":
    handler.handle()
