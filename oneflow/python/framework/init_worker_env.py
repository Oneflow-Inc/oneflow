from __future__ import absolute_import

import os
import sys
import getpass
import google.protobuf.text_format as pbtxt
from tempfile import NamedTemporaryFile
from oneflow.core.job.job_set_pb2 import ConfigProto
import oneflow.core.job.job_set_pb2 as job_set_util

def SystemCall(cmd):
  print(cmd)
  os.system(cmd)

def SendBinaryAndConfig2Worker(machine, oneflow_worker_path, config_proto_path, run_dir):
    ssh_prefix = "ssh " + getpass.getuser() + "@" + machine.addr + " "
    remote_file_prefix = " " + getpass.getuser() + "@" + machine.addr + ":"
    SystemCall(ssh_prefix + "\"mkdir -p " + run_dir + "\"")
    assert(run_dir != "")
    SystemCall(ssh_prefix + "\"rm -rf " + run_dir + "/*\"")
    SystemCall("scp " + oneflow_worker_path + remote_file_prefix + run_dir + "/oneflow_worker")
    SystemCall("scp " + config_proto_path + remote_file_prefix + run_dir + "/config.proto")
    oneflow_cmd = "\"cd " + run_dir + "; " \
                + "nohup ./oneflow_worker -logtostderr=0 -log_dir=./log -v=0 -logbuflevel=-1 " \
                + "-config_proto=./config.proto "  \
                + " 1>/dev/null 2>&1 </dev/null & \""
    SystemCall(ssh_prefix + oneflow_cmd);


def TryInitOneflowWorkerEnv(config_proto):
    assert(type(config_proto) is job_set_util.ConfigProto)
    #config_proto_str = pbtxt.MessageToString(config_proto)
    resource = config_proto.resource
    assert(len(resource.machine) > 0)
    oneflow_worker_path = os.getenv("ONEFLOW_WORKER_BIN")
    run_dir = os.getenv("HOME") + "/oneflow_worker_run_dir"
    oneflow_worker_path = os.path.abspath(os.path.expanduser(oneflow_worker_path))
    assert os.path.isfile(oneflow_worker_path)
    run_dir = os.path.abspath(os.path.expanduser(run_dir))
    config_file = NamedTemporaryFile(delete=False)
    config_file.write(pbtxt.MessageToString(config_proto))
    config_file.close()

    for machine in resource.machine:
        if machine.id == 0:
            continue
        SendBinaryAndConfig2Worker(machine, oneflow_worker_path, config_file.name, run_dir)

    os.remove(config_file.name)
