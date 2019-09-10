#include "oneflow/core/control/cluster_control.h"
#include "oneflow/core/control/cluster_control.pb.h"
#include "oneflow/core/control/ctrl_server.h"
#include "oneflow/core/control/ctrl_client.h"
#include "oneflow/core/job/machine_context.h"
#include "oneflow/core/job/resource_desc.h"

namespace oneflow {

namespace {

std::string GetHaltAckCtrlKey(int64_t machine_id) {
  return "HaltAckCtrlKey/" + std::to_string(machine_id);
}

// return unique sequential key
// because ctrl key is not allowed to push/pull twice
std::string GetHaltOrSessionStartCtrlKey() {
  static int64_t seq = 0;
  return "HaltOrSessionStart/" + std::to_string(seq++);
}

void MasterWaitHaltAck() {
  FOR_RANGE(int64_t, i, 0, Global<ResourceDesc>::Get()->TotalMachineNum()) {
    if (i == Global<MachineCtx>::Get()->this_machine_id()) { continue; }
    ClusterControlProto cluster_control_proto;
    Global<CtrlClient>::Get()->PullKV(GetHaltAckCtrlKey(i), &cluster_control_proto);
  }
}

}  // namespace

void ClusterControl::MasterSendSessionStart() {
  ClusterControlProto cluster_control;
  cluster_control.set_cmd(kClusterCtrlCmdSessionStart);
  Global<CtrlClient>::Get()->PushKV(GetHaltOrSessionStartCtrlKey(), cluster_control);
}

void ClusterControl::MasterSendHaltAndWaitAck() {
  ClusterControlProto cluster_control;
  cluster_control.set_cmd(kClusterCtrlCmdHalt);
  Global<CtrlClient>::Get()->PushKV(GetHaltOrSessionStartCtrlKey(), cluster_control);
  MasterWaitHaltAck();
}

bool ClusterControl::WorkerReceiveHalt() {
  ClusterControlProto cluster_control;
  Global<CtrlClient>::Get()->PullKV(GetHaltOrSessionStartCtrlKey(), &cluster_control);
  if (cluster_control.cmd() == kClusterCtrlCmdHalt) { return true; }
  CHECK_EQ(cluster_control.cmd(), kClusterCtrlCmdSessionStart);
  return false;
}

void ClusterControl::WorkerSendHaltAck() {
  const auto& machine_id_key = GetHaltAckCtrlKey(Global<MachineCtx>::Get()->this_machine_id());
  ClusterControlProto cluster_control_proto;
  cluster_control_proto.set_cmd(kClusterCtrlCmdHaltAck);
  Global<CtrlClient>::Get()->PushKV(machine_id_key, cluster_control_proto);
}

}  // namespace oneflow
