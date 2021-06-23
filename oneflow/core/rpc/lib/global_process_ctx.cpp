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
#include "oneflow/core/common/global.h"
#include "oneflow/core/common/str_util.h"
#include "oneflow/core/rpc/include/global_process_ctx.h"

namespace oneflow {

namespace {

HashMap<int64_t, int64_t> GetRank2NodeId(const NumProcessDistribution& num_process_distribution) {
  HashMap<int64_t, int64_t> rank2node_id;
  int64_t rank_offset = 0;
  for (int64_t node_id = 0; node_id < num_process_distribution.num_process_size(); ++node_id) {
    for (int16_t rank = 0; rank < num_process_distribution.num_process(node_id); ++rank) {
      CHECK(rank2node_id.emplace(rank + rank_offset, node_id).second);
    }
    rank_offset += num_process_distribution.num_process(node_id);
  }
  return rank2node_id;
}

HashMap<int64_t, int64_t> GetNodeId2RankOffset(
    const NumProcessDistribution& num_process_distribution) {
  HashMap<int64_t, int64_t> node_id2rankoffset;
  int64_t rank_offset = 0;
  for (int64_t node_id = 0; node_id < num_process_distribution.num_process_size(); ++node_id) {
    CHECK(node_id2rankoffset.emplace(node_id, rank_offset).second);
    rank_offset += num_process_distribution.num_process(node_id);
  }
  return node_id2rankoffset;
}

}  // namespace

void GlobalProcessCtx::GetCurrentMachineIdAndDeviceId(int64_t* machine_id, int64_t* device_id) {
  *machine_id = Rank();
  int64_t node_id = ThisNodeId();
  static HashMap<int64_t, int64_t> node_id2rankoffset =
      GetNodeId2RankOffset(NumProcessDistributionInCluster());
  int64_t rank_offset = node_id2rankoffset.at(node_id);
  *device_id = *machine_id - rank_offset;
}

int64_t GlobalProcessCtx::Rank() {
  CHECK_NOTNULL(Global<ProcessCtx>::Get());
  return Global<ProcessCtx>::Get()->rank();
}

int64_t GlobalProcessCtx::NodeSize() {
  CHECK_NOTNULL(Global<ProcessCtx>::Get());
  return Global<ProcessCtx>::Get()->node_size();
}

int64_t GlobalProcessCtx::ThisNodeId() {
  CHECK_NOTNULL(Global<ProcessCtx>::Get());
  return NodeId4Rank(Rank());
}

int64_t GlobalProcessCtx::NodeId4Rank(int64_t rank) {
  CHECK_NOTNULL(Global<ProcessCtx>::Get());
  static HashMap<int64_t, int64_t> rank2node_id = GetRank2NodeId(NumProcessDistributionInCluster());
  CHECK(rank2node_id.find(rank) != rank2node_id.end());
  return rank2node_id.at(rank);
}

int64_t GlobalProcessCtx::NumOfProcessOnNode() {
  if (Global<NumProcessDistribution>::Get() != nullptr) {
    return int64_t(Global<NumProcessDistribution>::Get()->num_process(0));
  }
  CHECK_NOTNULL(Global<ProcessCtx>::Get());
  int64_t node_id = ThisNodeId();
  return Global<ProcessCtx>::Get()->num_process_distribution_in_cluster().num_process(node_id);
}

const NumProcessDistribution& GlobalProcessCtx::NumProcessDistributionInCluster() {
  if (Global<NumProcessDistribution>::Get() != nullptr) {
    return *Global<NumProcessDistribution>::Get();
  }
  CHECK_NOTNULL(Global<ProcessCtx>::Get());
  return Global<ProcessCtx>::Get()->num_process_distribution_in_cluster();
}

bool GlobalProcessCtx::IsThisProcessMaster() {
  CHECK_NOTNULL(Global<ProcessCtx>::Get());
  return Global<ProcessCtx>::Get()->rank() == 0;
}

size_t GlobalProcessCtx::WorldSize() {
  CHECK_NOTNULL(Global<ProcessCtx>::Get());
  return Global<ProcessCtx>::Get()->ctrl_addr().size();
}

std::string GlobalProcessCtx::LogDirEntry() {
  CHECK_NOTNULL(Global<ProcessCtx>::Get());
  const auto& process_ctx = *Global<ProcessCtx>::Get();
  const auto& addr = process_ctx.ctrl_addr(process_ctx.rank());
  CHECK(addr.has_host());
  return addr.host() + "-" + std::to_string(addr.port()) + "-" + std::to_string(process_ctx.rank());
}

}  // namespace oneflow
