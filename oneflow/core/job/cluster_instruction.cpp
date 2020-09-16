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

std::string GetHaltAckCtrlKey(int64_t machine_id) {
  return "HaltAckCtrlKey/" + std::to_string(machine_id);
}

// return unique sequential key
// because ctrl key is not allowed to push/pull twice
std::string GetClusterInstructionKey() {
  static int64_t seq = 0;
  return "ClusterInstructionKey/" + std::to_string(seq++);
}

void OccasionallyClearCtrlKV() {
  static int64_t seq = 0;
  const static int64_t interval = 65536;
  // 1 instead of 0 is better for avoid clearing no ctrl kv
  if ((seq++) % interval == 1) {
    OF_BARRIER_ALL();
    Global<CtrlClient>::Get()->Clear();
    OF_BARRIER_ALL();
  }
}

void PushClusterInstruction(const ClusterInstructionProto& cluster_instruction) {
  OccasionallyClearCtrlKV();
  Global<CtrlClient>::Get()->PushKV(GetClusterInstructionKey(), cluster_instruction);
}

void PullClusterInstruction(ClusterInstructionProto* cluster_instruction) {
  OccasionallyClearCtrlKV();
  Global<CtrlClient>::Get()->PullKV(GetClusterInstructionKey(), cluster_instruction);
}

}  // namespace

void ClusterInstruction::NewSessionBarrier() {
  OF_BARRIER_ALL();
  Global<CtrlClient>::Get()->Clear();
  OF_BARRIER_ALL();
}

void ClusterInstruction::MasterSendSessionStart() {
  ClusterInstructionProto cluster_instruction;
  cluster_instruction.mutable_cluster_ctrl_session_start();
  PushClusterInstruction(cluster_instruction);
  NewSessionBarrier();
}

void ClusterInstruction::MasterSendHalt() {
  ClusterInstructionProto cluster_instruction;
  cluster_instruction.mutable_cluster_ctrl_halt();
  PushClusterInstruction(cluster_instruction);
  HaltBarrier();
}

void ClusterInstruction::MasterSendEagerInstruction(
    const ClusterInstructionProto& cluster_instruction) {
  CHECK(cluster_instruction.has_eager_instruction());
  PushClusterInstruction(cluster_instruction);
}

void ClusterInstruction::WorkerReceiveInstruction(ClusterInstructionProto* cluster_instruction) {
  PullClusterInstruction(cluster_instruction);
}

void ClusterInstruction::HaltBarrier() { OF_BARRIER_ALL(); }

}  // namespace oneflow
