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
import oneflow
import oneflow._oneflow_internal
import oneflow.core.job.job_set_pb2 as job_set_util

import inspect
import enum
from abc import ABC, abstractmethod


class Session(ABC):
    class SessionStatus(enum.Enum):
        OPEN = 1
        RUNNING = 2
        CLOSED = 3

    def __init__(self, sess_id):
        self.config_proto_ = None
        self.resource_ = None

        self.sess_ = oneflow._oneflow_internal.RegsiterSession(sess_id)
        self.status_ = self.SessionStatus.OPEN

    @property
    def id(self):
        return self.sess_.id

    @property
    def status(self):
        return self.status_

    @property
    def is_running(self):
        return self.status_ == self.SessionStatus.RUNNING

    @property
    def config_proto(self):
        if self.config_proto_ is None:
            self.config_proto_ = _GetDefaultConfigProto()
        return self.config_proto_

    @property
    def resource(self):
        if self.resource_ is None:
            return oneflow.env.current_resource()
        else:
            return self.resource_

    def _check_status(self, *status):
        check_success = False
        for stat in status:
            if self.status_ == stat:
                check_success = True
                break

        if check_success is False:
            caller_func_name = inspect.stack()[1].function
            allowed_status = ",".join(status)
            raise ValueError(
                "The calling to {} is only allowed when status is {}, current status is {}".format(
                    caller_func_name, allowed_status, self.status_
                )
            )

    def Init(self):
        self._check_status(self.SessionStatus.OPEN)

        if not oneflow._oneflow_internal.IsEnvInited():
            oneflow.env.init()

        _TryCompleteConfigProto(self.config_proto)
        self.resource_ = self.config_proto.resource

        self.init()

        self.status_ = self.SessionStatus.RUNNING
        return self

    def TryInit(self):
        if self.status_ == self.SessionStatus.OPEN:
            self.Init()
        return self

    @abstractmethod
    def init(self):
        pass

    def TryClose(self):
        if self.status_ == self.SessionStatus.RUNNING:
            self.Close()

    def Close(self):
        self._check_status(self.SessionStatus.RUNNING)
        self.close()
        self.resource_ = None
        self.status_ = self.SessionStatus.CLOSED

    @abstractmethod
    def close(self):
        pass

    def has_empty_is_mirrored_strategy_enabled_stack(self):
        return self.sess_.is_mirrored_strategy_enabled_stack_size() == 0

    def push_mirrored_strategy_enabled(self, val):
        assert isinstance(val, bool)
        self.sess_.push_mirrored_strategy_enabled(val)

    def pop_mirrored_strategy_enabled(self):
        self.sess_.pop_mirrored_strategy_enabled()

    def is_mirrored_strategy_enabled(self):
        return self.sess_.is_mirrored_strategy_enabled()

    def is_consistent_strategy_enabled(self):
        return self.sess_.is_consistent_strategy_enabled()


def _TryCompleteConfigProto(config_proto):
    if config_proto.resource.machine_num == 0:
        config_proto.resource.machine_num = oneflow._oneflow_internal.GetNodeSize()


def _GetDefaultConfigProto():
    config_proto = job_set_util.ConfigProto()
    config_proto.resource.machine_num = 0
    if oneflow._oneflow_internal.flags.with_cuda():
        config_proto.resource.gpu_device_num = 1
    else:
        raise TypeError
