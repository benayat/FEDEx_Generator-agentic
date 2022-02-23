
class Operation:
    def __init__(self, scheme):
        self.scheme = scheme
        self.bins_count = 500

    def set_bins_count(self, n):
        self.bins_count = n

    def get_bins_count(self):
        return self.bins_count

    def calc_measures(self):
        raise NotImplementedError()
