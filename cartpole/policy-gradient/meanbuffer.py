
class MeanBuffer:
    def __init__(self, cap):
        self.cap = cap
        self.deque = collections.deque(maxlen=self.cap)
        self.sum = 0.0

    def push(self, val):
        if len(self.deque) == self.cap:
            self.sum -= self.deque[0]
        self.deque.append(val)
        self.sum += val

    def mean(self):
        if not self.deque:
            return 0.0

        return self.sum / len(self.deque)