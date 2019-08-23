from __future__ import absolute_import

import uuid
import os
import sys
import getpass
import google.protobuf.text_format as pbtxt
from tempfile import NamedTemporaryFile
import oneflow.core.job.job_set_pb2 as job_set_util
from oneflow.core.job.job_set_pb2 import ConfigProto
from oneflow.python.oneflow_export import oneflow_export
import oneflow.python.framework.config_util as config_util

@oneflow_export('deprecated.init_worker_and_master')
def init_worker_and_master(config_proto):
    import oneflow
    oneflow.deprecated.init_worker(config_proto)
    oneflow.init(config_proto)

@oneflow_export('deprecated.init_worker')
def init_worker(config_proto):
    if (isinstance(config_proto, config_util.ConfigProtoBuilder)):
        config_proto = config_proto.config_proto
    assert isinstance(config_proto, ConfigProto)
    config_util.TryCompleteDefaultConfigProto(config_proto)
    assert(type(config_proto) is job_set_util.ConfigProto)
    resource = config_proto.resource
    assert(len(resource.machine) > 0)
    oneflow_worker_path = os.getenv("ONEFLOW_WORKER_BIN")
    global _temp_run_dir
    _temp_run_dir = os.getenv("HOME") + "/oneflow_temp/" + str(uuid.uuid1())
    run_dir = _temp_run_dir
    assert os.path.isfile(oneflow_worker_path)
    run_dir = os.path.abspath(os.path.expanduser(run_dir))
    config_file = NamedTemporaryFile(delete=False)
    config_file.write(pbtxt.MessageToString(config_proto))
    config_file.close()

    for machine in resource.machine:
        if machine.id == 0:
            continue
        _SendBinaryAndConfig2Worker(machine, oneflow_worker_path, config_file.name)

    os.remove(config_file.name)


@oneflow_export('deprecated.delete_worker')
def delete_worker(config_proto):
    if (isinstance(config_proto, config_util.ConfigProtoBuilder)):
        config_proto = config_proto.config_proto
    assert isinstance(config_proto, ConfigProto)
    assert(type(config_proto) is job_set_util.ConfigProto)
    global _temp_run_dir
    assert(_temp_run_dir != "")
    for machine in config_proto.resource.machine:
        if machine.id == 0:
            continue
        _SystemCall(ssh_prefix + "\"rm -r " + _temp_run_dir + "\"")

def _SendBinaryAndConfig2Worker(machine, oneflow_worker_path, config_proto_path):
    global _temp_run_dir
    run_dir = _temp_run_dir
    ssh_prefix = "ssh " + getpass.getuser() + "@" + machine.addr + " "
    remote_file_prefix = " " + getpass.getuser() + "@" + machine.addr + ":"
    assert(run_dir != "")
    _SystemCall(ssh_prefix + "\"mkdir -p " + run_dir + "\"")
    _SystemCall("scp " + oneflow_worker_path + remote_file_prefix + run_dir + "/oneflow_worker")
    _SystemCall("scp " + config_proto_path + remote_file_prefix + run_dir + "/config.proto")
    oneflow_cmd = "\"cd " + run_dir + "; " \
                + "nohup ./oneflow_worker -logtostderr=0 -log_dir=./log -v=0 -logbuflevel=-1 " \
                + "-config_proto=./config.proto "  \
                + " 1>/dev/null 2>&1 </dev/null & \""
    _SystemCall(ssh_prefix + oneflow_cmd);

def _SystemCall(cmd):
  print(cmd)
  os.system(cmd)

_temp_run_dir = ""

