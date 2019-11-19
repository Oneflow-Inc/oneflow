from __future__ import absolute_import

import uuid
import os
import sys
import getpass
import google.protobuf.text_format as pbtxt
from tempfile import NamedTemporaryFile
from oneflow.core.job.cluster_pb2 import ClusterProto
from oneflow.python.oneflow_export import oneflow_export
import oneflow.python.framework.cluster_util as cluster_util

@oneflow_export('deprecated.init_worker')
def init_worker(scp_binary = True, use_uuid = True):
    assert type(cluster_util.default_cluster_proto) is ClusterProto
    cluster_util.defautl_cluster_proto_mutable = False
    cluster_proto = cluster_util.default_cluster_proto
    assert len(cluster_proto.machine) > 0
    assert type(scp_binary) is bool
    assert type(use_uuid) is bool
    oneflow_worker_path = os.getenv("ONEFLOW_WORKER_BIN")
    assert oneflow_worker_path is not None, "please set env ONEFLOW_WORKER_BIN"
    assert os.path.isfile(
        oneflow_worker_path), "binary oneflow_worker not found, please check your environment variable ONEFLOW_WORKER_BIN, path: {}".format(oneflow_worker_path)
    global _temp_run_dir
    if use_uuid:
        assert scp_binary is True
        _temp_run_dir = os.getenv("HOME") + "/oneflow_temp/" + str(uuid.uuid1())
    else:
        _temp_run_dir = os.getenv("HOME") + "/oneflow_temp/no_uuid"
    run_dir = _temp_run_dir
    run_dir = os.path.abspath(os.path.expanduser(run_dir))
    cluster_file = NamedTemporaryFile(delete=False)
    if sys.version_info >= (3, 0):
        cluster_file.write(pbtxt.MessageToString(cluster_proto).encode())
    else:
        cluster_file.write(pbtxt.MessageToString(cluster_proto))
    cluster_file.close()

    for machine in cluster_proto.machine:
        if machine.id == 0:
            continue
        _SendBinaryAndConfig2Worker(machine, oneflow_worker_path, cluster_file.name, run_dir, scp_binary)

    os.remove(cluster_file.name)


@oneflow_export('deprecated.delete_worker')
def delete_worker():
    assert cluster_util.cluster_proto_mutable == False
    cluster_proto = cluster_util.default_cluster_proto
    assert isinstance(cluster_proto, ClusterProto)
    global _temp_run_dir
    assert _temp_run_dir != ""
    for machine in cluster_proto.machine:
        if machine.id == 0:
            continue
        ssh_prefix = "ssh " + getpass.getuser() + "@" + machine.addr + " "
        _SystemCall(ssh_prefix + "\"rm -r " + _temp_run_dir + "\"")

def _SendBinaryAndConfig2Worker(machine, oneflow_worker_path, cluster_proto_path, run_dir, scp_binary):
    _SystemCall("ssh-copy-id -f " + getpass.getuser() + "@" + machine.addr)
    ssh_prefix = "ssh " + getpass.getuser() + "@" + machine.addr + " "
    remote_file_prefix = " " + getpass.getuser() + "@" + machine.addr + ":"
    assert run_dir != ""
    _SystemCall(ssh_prefix + "\"mkdir -p " + run_dir + "\"")
    if scp_binary:
        _SystemCall("scp " + oneflow_worker_path + remote_file_prefix + run_dir + "/oneflow_worker")
    _SystemCall("scp " + cluster_proto_path + remote_file_prefix + run_dir + "/cluster.proto")
    oneflow_cmd = "\"cd " + run_dir + "; " \
                + "nohup ./oneflow_worker -logtostderr=0 -log_dir=./log -v=0 -logbuflevel=-1 " \
                + "-cluster_proto=./cluster.proto "  \
                + " 1>/dev/null 2>&1 </dev/null & \""
    _SystemCall(ssh_prefix + oneflow_cmd);

def _SystemCall(cmd):
  print(cmd)
  os.system(cmd)

_temp_run_dir = ""
