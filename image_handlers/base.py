from multiprocessing import Process, Queue
from queue import Empty

import cv2


class BaseImageHandler:
    def __init__(
        self,
        input_stream=0,
        window_title="cam process",
        max_buffer_size=1,
        processors=(),
    ):
        self.input_stream = input_stream
        self.cap = cv2.VideoCapture(input_stream)
        # self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        # self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.window_title = window_title
        self.buffer = Queue(maxsize=max_buffer_size)
        self.processors = processors
        self.current_state = None

    def handle(self):
        while self.cap.isOpened():
            success, image = self.cap.read()

            if not success:
                break

            yield image

    def read_input(self):
        for data in self.handle():
            self.current_state = data
            if not self.buffer.full():
                self.buffer.put(data, False)

    def read_next_data(self):
        try:
            data = self.buffer.get()
        except Empty:
            data = None

        return data

    def implement_processors(self):
        for processor in self.processors:
            self.current_state = processor(self.current_state)
            # if str(processor):
            #     print(processor)


class BaseImageHandlerProcess(BaseImageHandler):
    def __init__(
        self,
        input_stream=0,
        window_title="cam process",
        max_buffer_size=1,
        processors=(),
    ):
        super().__init__(
            input_stream, window_title, max_buffer_size, processors
        )
        self.process = Process()

    def start(self):
        self.process = Process(target=self.read_input)
        self.process.start()

    def stop(self):
        self.cap.release()
        self.process.terminate()
