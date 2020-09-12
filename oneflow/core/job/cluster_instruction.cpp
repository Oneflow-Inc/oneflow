/*
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
*/
#include "oneflow/core/job/cluster_instruction.h"
#include "oneflow/core/job/cluster_instruction.pb.h"
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

void ClusterInstruction::MasterSendSessionStart() {
  BarrierClear();
  ClusterInstructionProto cluster_instruction;
  cluster_instruction.mutable_cluster_ctrl_session_start();
  Global<CtrlClient>::Get()->PushKV(GetHaltOrSessionStartCtrlKey(), cluster_instruction);
}

void ClusterInstruction::MasterSendHalt() {
  BarrierClear();
  ClusterInstructionProto cluster_instruction;
  cluster_instruction.mutable_cluster_ctrl_halt();
  Global<CtrlClient>::Get()->PushKV(GetHaltOrSessionStartCtrlKey(), cluster_instruction);
  HaltBarrier();
}

bool ClusterInstruction::WorkerReceiveHalt(ClusterInstructionProto* cluster_instruction) {
  BarrierClear();
  Global<CtrlClient>::Get()->PullKV(GetHaltOrSessionStartCtrlKey(), cluster_instruction);
  if (cluster_instruction->has_cluster_ctrl_halt()) { return true; }
  CHECK(cluster_instruction->has_cluster_ctrl_session_start());
  return false;
}

void ClusterInstruction::HaltBarrier() { OF_BARRIER_ALL(); }

}  // namespace oneflow
