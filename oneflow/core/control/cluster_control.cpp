#include "oneflow/core/control/cluster_control.h"
#include "oneflow/core/control/cluster_control.pb.h"
#include "oneflow/core/control/ctrl_server.h"
#include "oneflow/core/control/ctrl_client.h"
#include "oneflow/core/job/machine_context.h"
#include "oneflow/core/job/env_desc.h"

namespace oneflow {

namespace {

void BarrierClear() {
  OF_BARRIER_ALL();
  Global<CtrlClient>::Get()->Clear();
  OF_BARRIER_ALL();
}

std::string GetHaltAckCtrlKey(int64_t machine_id) {
  return "HaltAckCtrlKey/" + std::to_string(machine_id);
}

// return unique sequential key
// because ctrl key is not allowed to push/pull twice
std::string GetHaltOrSessionStartCtrlKey() {
  static int64_t seq = 0;
  return "HaltOrSessionStart/" + std::to_string(seq++);
}

}  // namespace

void ClusterControl::MasterSendSessionStart() {
  BarrierClear();
  ClusterControlProto cluster_control;
  cluster_control.set_cmd(kClusterCtrlCmdSessionStart);
  Global<CtrlClient>::Get()->PushKV(GetHaltOrSessionStartCtrlKey(), cluster_control);
}

void ClusterControl::MasterSendHalt() {
  BarrierClear();
  ClusterControlProto cluster_control;
  cluster_control.set_cmd(kClusterCtrlCmdHalt);
  Global<CtrlClient>::Get()->PushKV(GetHaltOrSessionStartCtrlKey(), cluster_control);
  HaltBarrier();
}

bool ClusterControl::WorkerReceiveHalt() {
  BarrierClear();
  ClusterControlProto cluster_control;
  Global<CtrlClient>::Get()->PullKV(GetHaltOrSessionStartCtrlKey(), &cluster_control);
  if (cluster_control.cmd() == kClusterCtrlCmdHalt) { return true; }
  CHECK_EQ(cluster_control.cmd(), kClusterCtrlCmdSessionStart);
  return false;
}

void ClusterControl::HaltBarrier() { OF_BARRIER_ALL(); }

}  // namespace oneflow
