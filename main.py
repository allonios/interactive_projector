from image_handler import ImageHandler

tracker = ImageHandler(min_detection_confidence=0.9)

if __name__ == "__main__":
    tracker.track_hands()