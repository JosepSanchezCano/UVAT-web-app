from threading import Timer

class Videoplayer():
    def __init__(self, interval, function):
        self._timer = None
        self.interval = interval
        self.function = function
        self.is_running = False
        self.is_forward = True
        self.playing = False

    def _run(self):

        self.is_running = False
        self.start()
        if self.is_forward:
            self.function(self.is_forward)
        else:
            self.function(self.is_forward)

    def start(self):
        if not self.is_running:

            self._timer = Timer(self.interval, self._run)            
            self._timer.start()
            self.is_running = True
            self.playing = True

    def stop(self):
        self.is_forward = True
        self._timer.cancel()
        self.is_running = False
        self.playing = False

    def change_forward(self):
        self.is_forward = not self.is_forward