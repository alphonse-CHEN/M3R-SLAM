import queue
import threading


def try_get_msg(q):
    try:
        msg = q.get_nowait()
    except queue.Empty:
        msg = None
    return msg


class FakeQueue:
    def put(self, arg):
        del arg

    def get_nowait(self):
        raise queue.Empty

    def qsize(self):
        return 0

    def empty(self):
        return True


def new_queue(manager, use_fake=False):
    if use_fake:
        return FakeQueue()
    return manager.Queue()


class FakeValue:
    """Drop-in replacement for mp.Manager().Value() using a plain attribute."""
    def __init__(self, typecode, initial):
        self.value = initial


class FakeManager:
    """Drop-in replacement for mp.Manager() that uses plain Python objects.

    Avoids multiprocessing overhead and Windows shared-memory issues
    when running in single-thread mode.
    """

    def RLock(self):
        return threading.RLock()

    def Lock(self):
        return threading.Lock()

    def Value(self, typecode, initial):
        return FakeValue(typecode, initial)

    def list(self, *args):
        return list(*args)

    def Queue(self):
        return FakeQueue()

    def shutdown(self):
        pass
