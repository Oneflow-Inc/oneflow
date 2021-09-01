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
import getpass
import os
import subprocess
import sys
import uuid
from tempfile import NamedTemporaryFile

from google.protobuf import text_format as pbtxt

from oneflow.compatible.single_client.framework import env_util as env_util
from oneflow.core.control.ctrl_bootstrap_pb2 import BootstrapConf
from oneflow.core.job.env_pb2 import EnvProto


def delete_worker_by_bootstrap(ssh_port=22) -> None:
    ssh_port_arg = " -p {} ".format(ssh_port)
    bootstrap_conf_list = env_util.global_ctrl_bootstrap_confs
    assert isinstance(bootstrap_conf_list, list)
    global _temp_run_dir
    assert _temp_run_dir != ""
    for bootstrap_conf in bootstrap_conf_list:
        assert isinstance(bootstrap_conf, BootstrapConf)
        if bootstrap_conf.rank == 0:
            continue
        ssh_prefix = (
            "ssh {} ".format(ssh_port_arg)
            + getpass.getuser()
            + "@"
            + bootstrap_conf.host
            + " "
        )
        if os.getenv("ONEFLOW_WORKER_KEEP_LOG"):
            print("worker log kept at: {}".format(bootstrap_conf.host), flush=True)
        else:
            _SystemCall(ssh_prefix + '"rm -r ' + _temp_run_dir + '"')
            print("temp run dir removed at: {}".format(bootstrap_conf.host), flush=True)


def delete_worker_of_multi_process(run_dir) -> None:
    assert run_dir != ""
    if os.getenv("ONEFLOW_WORKER_KEEP_LOG"):
        print("worker log kept at localhost:" + run_dir, flush=True)
    else:
        os.system("rm -r " + run_dir)
        print("temp run dir removed at localhost:" + run_dir, flush=True)


_temp_run_dir = ""
