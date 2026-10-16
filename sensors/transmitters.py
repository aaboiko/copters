class Transmitter:
    def __init__(self, world):
        self.world = world


    def signal_on(self, pattern, id, pose):
        self.world.signal_on(pattern, id, pose)


    def signal_off(self, id):
        self.world.signal_off(id)


class UpperTransmitter(Transmitter):
    def __init__(self, world):
        super().__init__(world)


class LowerTransmitter(Transmitter):
    def __init__(self, world):
        super().__init__(world)