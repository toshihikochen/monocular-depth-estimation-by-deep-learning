import time


# time format
# show the time in the format of hh:mm:ss
def format_time(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return "%d:%02d:%02d" % (h, m, s)


class Timer:
    """
    A timer class that can recode the time interval between epoch start and epoch end,
    and can calculate the ETA of the training process.
    """

    def __init__(self):
        self.start_time = time.time()
        self.end_time = time.time()
        self.interval = 0
        self.last_eta = 0

    def start(self):
        self.start_time = time.time()

    def stop(self):
        self.end_time = time.time()
        self.interval = self.end_time - self.start_time
        return self.interval

    def reset(self):
        self.__init__()

    def eta(self, current_step, total_step, current_time=time.time()):
        self.last_eta = (current_time / current_step * (total_step - current_step)) * 0.95 + self.last_eta * 0.05
        return self.last_eta

    def verbose(self, current_step, total_step):
        current_time = time.time() - self.start_time
        return f"{format_time(current_time)}>{format_time(self.eta(current_step, total_step, current_time))}"

    def __call__(self, current_step, total_step):
        return self.verbose(current_step, total_step)
