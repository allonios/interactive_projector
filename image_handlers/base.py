from multiprocessing import Process, Queue
from queue import Empty

import cv2


class BaseImageHandlerProcess:
    def __init__(self, input_stream=0, window_title="cam process", max_buffer_size=1):
        self.process = Process()
        self.input_stream = input_stream
        self.cap = cv2.VideoCapture(input_stream)
        self.window_title = window_title
        self.buffer = Queue(maxsize=max_buffer_size)

    def handle(self):
        while self.cap.isOpened():
            success, image = self.cap.read()

            if not success:
                break

            yield image

    def read_input(self):
        for data in self.handle():
            if not self.buffer.full():
                self.buffer.put(data, False)

    def read_next_data(self):
        try:
            data = self.buffer.get()
        except Empty:
            data = None

        return data

    def start(self):
        self.process = Process(target=self.read_input)
        self.process.start()

    def stop(self):
        self.cap.release()
        self.process.terminate()
