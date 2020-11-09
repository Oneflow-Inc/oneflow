"""
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
from __future__ import absolute_import

import oneflow.python.framework.c_api_util as c_api_util
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


class ParallelDescSymbol(Symbol):
    def __init__(self, symbol_id, parallel_conf):
        Symbol.__init__(self, symbol_id, parallel_conf)
        self.device_tag_ = parallel_conf.device_tag
        self.machine_id2device_id_list_ = MakeMachineId2DeviceIdList(parallel_conf)
        sub_parallel_nums = [len(v) for k, v in self.machine_id2device_id_list_.items()]
        self.parallel_num_ = functools.reduce(lambda a, b: a + b, sub_parallel_nums, 0)
        self.hash_ = hash(self.device_tag_) ^ hash(str(self.machine_id2device_id_list_))

    def __hash__(self):
        return self.hash_

    def __eq__(lhs, rhs):
        return (
            lhs.device_tag_ == rhs.device_tag_
            and lhs.machine_id2device_id_list_ == rhs.machine_id2device_id_list_
        )

    def __str__(self):
        return str(self.parallel_conf)

    @property
    def parallel_conf(self):
        return self.data

    @property
    def parallel_num(self):
        return self.parallel_num_

    @property
    def device_tag(self):
        return self.device_tag_

    @property
    def machine_id2device_id_list(self):
        return self.machine_id2device_id_list_

    def Containing(self, other):
        if self.device_tag != other.device_tag:
            return False
        return _GlobalDeviceIdsContaining(
            self.machine_id2device_id_list, other.machine_id2device_id_list,
        )


def _GlobalDeviceIdsContaining(bigger, smaller):
    for machine_id, device_ids in smaller.items():
        if machine_id not in bigger:
            return False
        bigger_device_ids = bigger[machine_id]
        for device_id in device_ids:
            if device_id not in bigger_device_ids:
                return False
    return True


def MakeMachineId2DeviceIdList(parallel_conf):
    parallel_conf_str = parallel_conf.SerializeToString()
    global _parallel_conf_str2ofrecord
    if parallel_conf_str not in _parallel_conf_str2ofrecord:
        ofrecord = c_api_util.GetMachine2DeviceIdListOFRecordFromParallelConf(
            parallel_conf
        )
        _parallel_conf_str2ofrecord[parallel_conf_str] = {
            int(k): list(v.int32_list.value) for k, v in ofrecord.feature.items()
        }
    return _parallel_conf_str2ofrecord[parallel_conf_str]


_parallel_conf_str2ofrecord = {}
