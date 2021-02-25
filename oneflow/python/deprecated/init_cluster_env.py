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

import getpass
import os
import sys
import uuid
from tempfile import NamedTemporaryFile

import google.protobuf.text_format as pbtxt
import oneflow.python.framework.env_util as env_util
from oneflow.core.job.env_pb2 import EnvProto
from oneflow.core.control.ctrl_bootstrap_pb2 import BootstrapConf
from oneflow.python.oneflow_export import oneflow_export
import subprocess


@oneflow_export("deprecated.init_worker")
def init_worker(
    scp_binary: bool = True,
    use_uuid: bool = True,
    ssh_port=22,
    bootstrap_conf_list=None,
) -> None:
    assert type(env_util.default_env_proto) is EnvProto
    env_util.defautl_env_proto_mutable = False
    env_proto = env_util.default_env_proto
    assert len(env_proto.machine) > 0 or len(bootstrap_conf_list) > 0
    assert type(scp_binary) is bool
    assert type(use_uuid) is bool
    oneflow_worker_path = os.getenv("ONEFLOW_WORKER_BIN")
    assert oneflow_worker_path is not None, "please set env ONEFLOW_WORKER_BIN"
    assert os.path.isfile(
        oneflow_worker_path
    ), "binary oneflow_worker not found, please check your environment variable ONEFLOW_WORKER_BIN, path: {}".format(
        oneflow_worker_path
    )
    global _temp_run_dir
    if use_uuid:
        assert scp_binary is True
        _temp_run_dir = os.getenv("HOME") + "/oneflow_temp/" + str(uuid.uuid1())
    else:
        _temp_run_dir = os.getenv("HOME") + "/oneflow_temp/no_uuid"
    run_dir = _temp_run_dir
    run_dir = os.path.abspath(os.path.expanduser(run_dir))
    if bootstrap_conf_list is None:
        env_file = NamedTemporaryFile(delete=False)
        if sys.version_info >= (3, 0):
            env_file.write(pbtxt.MessageToString(env_proto).encode())
        else:
            env_file.write(pbtxt.MessageToString(env_proto))
        env_file.close()

        for machine in env_proto.machine:
            if machine.id == 0:
                pass
            else:
                _SendBinaryAndConfig2Worker(
                    machine.addr,
                    oneflow_worker_path,
                    env_file.name,
                    run_dir,
                    scp_binary,
                    ssh_port,
                )

        os.remove(env_file.name)
    else:
        worker_env_proto = EnvProto()
        worker_env_proto.CopyFrom(env_proto)
        for bootstrap_conf in bootstrap_conf_list:
            if bootstrap_conf.rank == 0:
                continue
            assert bootstrap_conf.HasField("host")
            worker_env_proto.ctrl_bootstrap_conf.CopyFrom(bootstrap_conf)
            env_file = NamedTemporaryFile(delete=False)
            if sys.version_info >= (3, 0):
                env_file.write(pbtxt.MessageToString(worker_env_proto).encode())
            else:
                env_file.write(pbtxt.MessageToString(worker_env_proto))
            env_file.close()
            _SendBinaryAndConfig2Worker(
                bootstrap_conf.host,
                oneflow_worker_path,
                env_file.name,
                run_dir,
                scp_binary,
                ssh_port,
            )
            os.remove(env_file.name)


@oneflow_export("deprecated.delete_worker")
def delete_worker(ssh_port=22) -> None:
    ssh_port_arg = " -p {} ".format(ssh_port)
    # assert env_util.env_proto_mutable == False
    env_proto = env_util.default_env_proto
    assert isinstance(env_proto, EnvProto)
    global _temp_run_dir
    assert _temp_run_dir != ""
    for machine in env_proto.machine:
        if machine.id == 0:
            continue
        ssh_prefix = (
            "ssh {} ".format(ssh_port_arg)
            + getpass.getuser()
            + "@"
            + machine.addr
            + " "
        )
        if os.getenv("ONEFLOW_WORKER_KEEP_LOG"):
            print("worker log kept at: {}".format(machine.addr), flush=True)
        else:
            _SystemCall(ssh_prefix + '"rm -r ' + _temp_run_dir + '"')
            print("temp run dir removed at: {}".format(machine.addr), flush=True)


@oneflow_export("deprecated.delete_worker_by_bootstrap")
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


def _SendBinaryAndConfig2Worker(
    addr, oneflow_worker_path, env_proto_path, run_dir, scp_binary, ssh_port
):
    ssh_port_arg = " -p {} ".format(ssh_port)
    scp_port_arg = " -P {} ".format(ssh_port)
    _SystemCall(
        "ssh-copy-id {} -f ".format(ssh_port_arg) + getpass.getuser() + "@" + addr
    )
    ssh_prefix = "ssh {}".format(ssh_port_arg) + getpass.getuser() + "@" + addr + " "
    remote_file_prefix = " " + getpass.getuser() + "@" + addr + ":"
    assert run_dir != ""
    _SystemCall(ssh_prefix + '"mkdir -p ' + run_dir + '"')
    if scp_binary:
        _SystemCall(
            "scp {}".format(scp_port_arg)
            + oneflow_worker_path
            + remote_file_prefix
            + run_dir
            + "/oneflow_worker"
        )
    _SystemCall(
        "scp {}".format(scp_port_arg)
        + env_proto_path
        + remote_file_prefix
        + run_dir
        + "/env.proto"
    )
    oneflow_cmd = (
        '"cd '
        + run_dir
        + "; "
        + "nohup ./oneflow_worker -logtostderr=0 -log_dir=./log -v=0 -logbuflevel=-1 "
        + "-env_proto=./env.proto "
        + ' 1>/dev/null 2>&1 </dev/null & "'
    )
    _SystemCall(ssh_prefix + oneflow_cmd)
    proc = subprocess.Popen(
        ssh_prefix + "ps aux",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        encoding="utf-8",
        shell=True,
    )
    outs, errs = proc.communicate(timeout=5)
    print(outs)
    assert "oneflow_worker" in str(outs), "fail to start oneflow_worker"
    print("oneflow worker initialized:", addr, flush=True)


def _SystemCall(cmd):
    print(cmd, flush=True)
    subprocess.check_call(cmd, shell=True)


_temp_run_dir = ""
