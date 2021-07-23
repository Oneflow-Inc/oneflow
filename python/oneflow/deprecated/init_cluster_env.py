import getpass
import os
import sys
import uuid
from tempfile import NamedTemporaryFile
import google.protobuf.text_format as pbtxt
import oneflow.framework.env_util as env_util
from oneflow.core.job.env_pb2 import EnvProto
from oneflow.core.control.ctrl_bootstrap_pb2 import BootstrapConf
import subprocess


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
