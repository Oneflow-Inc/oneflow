import oneflow.framework.c_api_util as c_api_util
import oneflow.core.job.placement_pb2 as placement_pb
import functools


class Symbol(object):
    def __init__(self, symbol_id, data):
        self.symbol_id_ = symbol_id
        self.data_ = data

    @property
    def symbol_id(self):
        return self.symbol_id_

    @property
    def data(self):
        return self.data_
