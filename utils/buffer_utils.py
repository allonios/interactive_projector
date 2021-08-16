from collections import deque
from multiprocessing.managers import BaseManager


class BufferManager(BaseManager):
    pass


class Empty(BaseException):
    pass


class Buffer:
    def __init__(self, initial_data=(), max_size=1):
        self.__data = deque(iterable=initial_data, maxlen=max_size)
        self.max_size = max_size

    def is_full(self) -> bool:
        return len(self.__data) >= self.max_size

    def add_item(self, item: object) -> None:
        if self.is_full():
            self.__data.popleft()
        self.__data.append(item)

    @property
    def latest_item(self):
        # print("status", not self.__data)
        if not self.__data:
            return None

        return self.__data[len(self.__data) - 1]

    @property
    def buffer_data(self):
        return self.__data

    def __len__(self):
        return len(self.__data)
