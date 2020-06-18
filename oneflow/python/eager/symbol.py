from __future__ import absolute_import

import oneflow.python.framework.placement_context as placement_ctx


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


class ParallelDescSymbol(Symbol):
    def __init__(self, symbol_id, parallel_conf, device_tag):
        Symbol.__init__(self, symbol_id, parallel_conf)
        self.device_tag_ = device_tag
        self.machine_id2device_id_list_ = placement_ctx.MakeMachineId2DeviceIdList(
            parallel_conf
        )
        self.hash_ = hash(self.device_tag_) ^ hash(str(self.machine_id2device_id_list_))

    def __hash__(self):
        return self.hash_

    def __eq__(lhs, rhs):
        return (
            lhs.device_tag_ == rhs.device_tag_
            and lhs.machine_id2device_id_list_ == rhs.machine_id2device_id_list_
        )

    @property
    def parallel_conf(self):
        return self.data

    @property
    def device_tag(self):
        return self.device_tag_

    @property
    def machine_id2device_id_list(self):
        return self.machine_id2device_id_list_
