from __future__ import absolute_import

import getpass
import os
import sys
import uuid
from tempfile import NamedTemporaryFile

import google.protobuf.text_format as pbtxt
import oneflow.python.framework.env_util as env_util
from oneflow.core.job.env_pb2 import EnvProto
from oneflow.python.oneflow_export import oneflow_export


@oneflow_export("deprecated.init_worker")
def init_worker(scp_binary=True, use_uuid=True):
    assert type(env_util.default_env_proto) is EnvProto
    env_util.defautl_env_proto_mutable = False
    env_proto = env_util.default_env_proto
    assert len(env_proto.machine) > 0
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
    env_file = NamedTemporaryFile(delete=False)
    if sys.version_info >= (3, 0):
        env_file.write(pbtxt.MessageToString(env_proto).encode())
    else:
        env_file.write(pbtxt.MessageToString(env_proto))
    env_file.close()

    for machine in env_proto.machine:
        if machine.id == 0:
            continue
        _SendBinaryAndConfig2Worker(
            machine, oneflow_worker_path, env_file.name, run_dir, scp_binary
        )

    os.remove(env_file.name)


@oneflow_export("deprecated.delete_worker")
def delete_worker():
    # assert env_util.env_proto_mutable == False
    env_proto = env_util.default_env_proto
    assert isinstance(env_proto, EnvProto)
    global _temp_run_dir
    assert _temp_run_dir != ""
    for machine in env_proto.machine:
        if machine.id == 0:
            continue
        ssh_prefix = "ssh " + getpass.getuser() + "@" + machine.addr + " "
        _SystemCall(ssh_prefix + '"rm -r ' + _temp_run_dir + '"')


def _SendBinaryAndConfig2Worker(
    machine, oneflow_worker_path, env_proto_path, run_dir, scp_binary
):
    _SystemCall("ssh-copy-id -f " + getpass.getuser() + "@" + machine.addr)
    ssh_prefix = "ssh " + getpass.getuser() + "@" + machine.addr + " "
    remote_file_prefix = " " + getpass.getuser() + "@" + machine.addr + ":"
    assert run_dir != ""
    _SystemCall(ssh_prefix + '"mkdir -p ' + run_dir + '"')
    if scp_binary:
        _SystemCall(
            "scp "
            + oneflow_worker_path
            + remote_file_prefix
            + run_dir
            + "/oneflow_worker"
        )
    _SystemCall("scp " + env_proto_path + remote_file_prefix + run_dir + "/env.proto")
    oneflow_cmd = (
        '"cd '
        + run_dir
        + "; "
        + "nohup ./oneflow_worker -logtostderr=0 -log_dir=./log -v=0 -logbuflevel=-1 "
        + "-env_proto=./env.proto "
        + ' 1>/dev/null 2>&1 </dev/null & "'
    )
    _SystemCall(ssh_prefix + oneflow_cmd)


def _SystemCall(cmd):
    print(cmd)
    os.system(cmd)


_temp_run_dir = ""
