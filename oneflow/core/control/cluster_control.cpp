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

void MasterWaitHaltAck() {
  FOR_RANGE(int64_t, i, 0, Global<ResourceDesc>::Get()->TotalMachineNum()) {
    ClusterControlProto cluster_control_proto;
    Global<CtrlClient>::Get()->PullKV(GetHaltAckCtrlKey(i), &cluster_control_proto);
  }
}

void WorkerSendAckAndExit() {
  const auto& machine_id_key = GetHaltAckCtrlKey(Global<MachineCtx>::Get()->this_machine_id());
  ClusterControlProto cluster_control_proto;
  cluster_control_proto.set_cmd(kClusterCtrlCmdHaltAck);
  Global<CtrlClient>::Get()->PushKV(machine_id_key, cluster_control_proto);
  exit(0);
}

}  // namespace

void ClusterControl::MasterSendSessionStart() {
  ClusterControlProto cluster_control;
  cluster_control.set_cmd(kClusterCtrlCmdSessionStart);
  Global<CtrlClient>::Get()->PushKV("halt_or_session_start", cluster_control);
}

void ClusterControl::MasterSendHaltAndWaitAck() {
  ClusterControlProto cluster_control;
  cluster_control.set_cmd(kClusterCtrlCmdHalt);
  Global<CtrlClient>::Get()->PushKV("session_end", cluster_control);
  MasterWaitHaltAck();
}

void ClusterControl::WorkerSendAckAndExitIfReceiveHalt() {
  ClusterControlProto cluster_control;
  Global<CtrlClient>::Get()->PullKV("halt_or_session_start", &cluster_control);
  if (cluster_control.cmd() == kClusterCtrlCmdHalt) { WorkerSendAckAndExit(); }
  CHECK_EQ(cluster_control.cmd(), kClusterCtrlCmdSessionStart);
}

}  // namespace oneflow
