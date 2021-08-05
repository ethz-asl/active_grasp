import time


class Timer:
    timers = dict()

    def __init__(self, name):
        self.name = name

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc_info):
        self.stop()

    def start(self):
        self.tic = time.perf_counter()

    def stop(self):
        elapsed_time = time.perf_counter() - self.tic
        self.timers[self.name] = elapsed_time
