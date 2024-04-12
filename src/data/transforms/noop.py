

class NoOp():
    """No operation transform."""

    def __call__(self, sample):
        return sample
