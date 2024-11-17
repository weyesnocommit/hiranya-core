from enum import unique, Enum
from multiprocessing import Queue, Process


class MLModelScheduler(object):
    def __init__(self):
        self._read_queue = Queue()
        self._write_queue = Queue()
        self._worker = None

    def start(self):
        self._worker.start()

    def shutdown(self):
        self._write_queue.put([MLWorkerCommands.SHUTDOWN, None])

    def _predict(self, *data, struct_temp):
        self._write_queue.put([MLWorkerCommands.PREDICT, data, struct_temp])
        return self._read_queue.get()

    def _train(self, *data):
        self._write_queue.put([MLWorkerCommands.TRAIN, data])
        return self._read_queue.get()

    def _save(self, *data):
        self._write_queue.put([MLWorkerCommands.SAVE, data])
        return self._read_queue.get()

    def _load(self, *data):
        self._write_queue.put([MLWorkerCommands.LOAD, data])
        return self._read_queue.get()


class MLModelWorker(Process):
    def __init__(self, name, read_queue: Queue, write_queue: Queue, use_gpu: bool):
        Process.__init__(self, name=name)
        self._read_queue = read_queue
        self._write_queue = write_queue
        self._use_gpu = use_gpu
        self._model = None

    def run(self):
        while True:
            vals = self._read_queue.get()
            command = vals[0]
            data = vals[1]
            if command == MLWorkerCommands.SHUTDOWN:
                return
            elif command == MLWorkerCommands.PREDICT:
                self._write_queue.put(self.predict(data, struct_temp=vals[2]))
            elif command == MLWorkerCommands.TRAIN:
                self._write_queue.put(self.train(data))
            elif command == MLWorkerCommands.SAVE:
                self._write_queue.put(self.save(data))
            elif command == MLWorkerCommands.LOAD:
                self._write_queue.put(self.load(data))

    def predict(self, *data):
        pass

    def train(self, *data):
        pass

    def save(self, *data):
        pass

    def load(self, *data):
        pass


@unique
class MLWorkerCommands(Enum):
    SHUTDOWN = 0
    TRAIN = 1
    PREDICT = 2
    SAVE = 3
    LOAD = 4