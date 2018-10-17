#!/usr/bin/env python

import os
import sys
import getpass
import google.protobuf.text_format as pbtxt
from tempfile import NamedTemporaryFile
from oneflow.core.job.dlnet_conf_pb2 import DLNetConf
from oneflow.core.job.resource_pb2 import Resource
from oneflow.core.job.placement_pb2 import Placement
from oneflow.core.job.job_conf_pb2 import JobConf1
from oneflow.core.job.job_conf_pb2 import JobConf2
from oneflow.core.job.job_conf_pb2 import OtherConf
    
def PrintHelpMessage():
  print "Usage: " + os.path.basename(sys.argv[0]) + " [option] /path/to/your.job"
  print """
  Options:
    --oneflow=oneflow_binary_filepath              default is ./oneflow
    --workdir=/path/to/workspace/on/all/machine    default is $HOME/oneflow_workdir
  """
  sys.exit(0)

def ParseJobConf1FromJobConf2(job_conf2_str):
  job_conf2 = pbtxt.Parse(job_conf2_str, JobConf2())
  job_conf = JobConf1();
  job_conf.net.CopyFrom(pbtxt.Parse(open(job_conf2.net).read(), DLNetConf()))
  job_conf.resource.CopyFrom(pbtxt.Parse(open(job_conf2.resource).read(), Resource()))
  job_conf.placement.CopyFrom(pbtxt.Parse(open(job_conf2.placement).read(), Placement()))
  job_conf.other.CopyFrom(pbtxt.Parse(open(job_conf2.other).read(), OtherConf()))
  return job_conf

def SystemCall(cmd):
  print cmd
  os.system(cmd)

def SubmitJobToOneMachine(machine, job_conf, oneflow_path, workdir):
  ssh_prefix = "ssh " + getpass.getuser() + "@" + machine.addr + " "
  remote_file_prefix = " " + getpass.getuser() + "@" + machine.addr + ":"
  SystemCall(ssh_prefix + "\"mkdir " + workdir + "\"")
  assert workdir != ""
  SystemCall(ssh_prefix + "\"rm -rf " + workdir + "/*\"")
  SystemCall("scp " + oneflow_path + remote_file_prefix + workdir)
  job_conf_file = NamedTemporaryFile(delete=False)
  job_conf_file.write(pbtxt.MessageToString(job_conf))
  job_conf_file.close()
  SystemCall("scp " + job_conf_file.name + remote_file_prefix + workdir + "/oneflow.job")
  os.remove(job_conf_file.name)
  oneflow_cmd = "bash -c \'\"cd " + workdir + "; nohup ./oneflow -logtostderr=0 -log_dir=./log -v=0 -logbuflevel=-1 " \
                + "-job_conf=./oneflow.job " \
                + " 1>/dev/null 2>&1 </dev/null & \"\'"
  SystemCall(ssh_prefix + oneflow_cmd);

def SubmitJob(job_conf, oneflow_path, workdir):
  for machine in job_conf.resource.machine:
    SubmitJobToOneMachine(machine, job_conf, oneflow_path, workdir)

def FindOneflowPathInEnv():
  for path_dir in os.getenv("PATH").split(":"):
    try:
      filenames = os.listdir(path_dir)
      if "oneflow" in filenames:
        return os.path.join(path_dir, "oneflow")
    except os.error:
      pass
  return ""

if __name__ == "__main__":
  if len(sys.argv) == 1:
    PrintHelpMessage()
  oneflow_path = "./oneflow"
  workdir = os.getenv("HOME") + "/oneflow_workdir"
  for arg in sys.argv[1:-1]:
    if arg.startswith("--oneflow="):
      oneflow_path = arg[10:]
    elif arg.startswith("--workdir="):
      workdir = arg[10:]
    else:
      PrintHelpMessage()
  oneflow_path = os.path.abspath(os.path.expanduser(oneflow_path))
  if os.path.isfile(oneflow_path) == False:
    oneflow_path = FindOneflowPathInEnv()
    assert os.path.isfile(oneflow_path)
  workdir = os.path.abspath(os.path.expanduser(workdir))
  job_conf_path = sys.argv[-1]
  if os.path.isfile(job_conf_path) == False:
    PrintHelpMessage()
  job_conf_str = open(job_conf_path).read();
  job_conf = JobConf1();
  try:
    job_conf = pbtxt.Parse(job_conf_str, job_conf)
  except pbtxt.ParseError:
    job_conf = ParseJobConf1FromJobConf2(job_conf_str)
  SubmitJob(job_conf, oneflow_path, workdir)
