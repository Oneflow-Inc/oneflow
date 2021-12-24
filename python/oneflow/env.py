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
from oneflow.framework.env_util import api_all_device_placement as all_device_placement
from oneflow.framework.env_util import api_ctrl_port as ctrl_port
from oneflow.framework.env_util import api_data_port as data_port
from oneflow.framework.env_util import api_env_init as init
from oneflow.framework.env_util import api_grpc_use_no_signal as grpc_use_no_signal
from oneflow.framework.env_util import api_init_bootstrap_confs as init_bootstrap_confs
from oneflow.framework.env_util import api_log_dir as log_dir
from oneflow.framework.env_util import api_logbuflevel as logbuflevel
from oneflow.framework.env_util import api_logtostderr as logtostderr
from oneflow.framework.env_util import api_machine as machine

import oneflow._oneflow_internal


def get_local_rank():
    """Returns the local rank of current machine.
    Local rank is not globally unique. It is only unique per process on a machine. 

    Returns:
        The the local rank of process on current machine.

    """
    return oneflow._oneflow_internal.GetLocalRank()


def get_rank():
    """Returns the rank of current process group.
    Rank is globally unique, range of which is [0, world_size). 

    Returns:
        The rank of the process group.

    """
    return oneflow._oneflow_internal.GetRank()


def get_node_size():
    """Returns the number of machines in the current process group.

    Returns:
        The the number of machines in the process group.

    """
    return oneflow._oneflow_internal.GetNodeSize()


def get_world_size():
    """Returns the number of processes in the current process group.

    Returns:
        The world size of the process group.

    """
    return oneflow._oneflow_internal.GetWorldSize()


def is_multi_client():
    """Returns whether it is currently in multi client mode.

    Returns:
        True if currently in multi client mode, otherwise returns Flase.

    """
    return oneflow._oneflow_internal.IsMultiClient()
