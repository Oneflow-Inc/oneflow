from __future__ import absolute_import

class Symbol(object):
    def __init__(self, symbol_id, data):
        self.symbol_id_ = symbol_id
        self.data_ = data

    @property
    def symbol_id(self): return self.symbol_id_

    @property
    def data(self): return self.data_

class ParallelDescSymbol(Symbol):
    def __init__(self, symbol_id, parallel_conf, device_tag):
        Symbol.__init__(self, symbol_id, parallel_conf)
        self.device_tag_ = device_tag

    @property
    def parallel_conf(self): return self.data

    @property
    def device_tag(self): return self.device_tag_
