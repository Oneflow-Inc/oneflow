class Lazy(object):
    def __init__(self, get_value):
        self.value_ = None
        self.has_value_ = False
        self.get_value_ = get_value

    @property
    def value(self):
        if not self.has_value_:
            self.value_ = self.get_value_()
            self.has_value_ = True
        return self.value_
