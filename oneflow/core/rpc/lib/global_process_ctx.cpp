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
#include "oneflow/core/common/singleton.h"
#include "oneflow/core/common/str_util.h"
#include "oneflow/core/control/ctrl_bootstrap.pb.h"
#include "oneflow/core/rpc/include/global_process_ctx.h"

namespace oneflow {

void GlobalProcessCtx::GetCurrentMachineIdAndDeviceId(int64_t* machine_id, int64_t* device_id) {
  *machine_id = Rank();
  *device_id = LocalRank();
}

int64_t GlobalProcessCtx::Rank() {
  CHECK_NOTNULL(Singleton<ProcessCtx>::Get());
  return Singleton<ProcessCtx>::Get()->rank();
}

int64_t GlobalProcessCtx::LocalRank() {
  char* local_rank_env = std::getenv("LOCAL_RANK");
  if (!local_rank_env) {
    static int64_t local_rank = Rank() % NumOfProcessPerNode();
    return local_rank;
  }
  CHECK(IsStrInt(local_rank_env));
  static int64_t local_rank = std::stol(local_rank_env);
  return local_rank;
}

int64_t GlobalProcessCtx::NodeSize() {
  CHECK_NOTNULL(Singleton<ProcessCtx>::Get());
  return Singleton<ProcessCtx>::Get()->node_size();
}

int64_t GlobalProcessCtx::ThisNodeId() {
  CHECK_NOTNULL(Singleton<ProcessCtx>::Get());
  return NodeId(Rank());
}

int64_t GlobalProcessCtx::NodeId(int64_t process_id) {
  CHECK_NOTNULL(Singleton<ProcessCtx>::Get());
  return process_id / NumOfProcessPerNode();
}

int64_t GlobalProcessCtx::NumOfProcessPerNode() {
  CHECK_NOTNULL(Singleton<ProcessCtx>::Get());
  CHECK_EQ(WorldSize() % NodeSize(), 0);
  return int64_t(WorldSize() / NodeSize());
}

bool GlobalProcessCtx::IsThisProcessMaster() {
  CHECK_NOTNULL(Singleton<ProcessCtx>::Get());
  return Singleton<ProcessCtx>::Get()->rank() == 0;
}

size_t GlobalProcessCtx::WorldSize() {
  CHECK_NOTNULL(Singleton<ProcessCtx>::Get());
  return Singleton<ProcessCtx>::Get()->ctrl_addr().size();
}

std::string GlobalProcessCtx::LogDirEntry() {
  CHECK_NOTNULL(Singleton<ProcessCtx>::Get());
  const auto& process_ctx = *Singleton<ProcessCtx>::Get();
  const auto& addr = process_ctx.ctrl_addr(process_ctx.rank());
  CHECK(addr.has_host());
  return addr.host() + "-" + std::to_string(addr.port()) + "-" + std::to_string(process_ctx.rank());
}

/* static */ int64_t GlobalProcessCtx::LocalRank(int64_t rank) {
  return rank % NumOfProcessPerNode();
}

}  // namespace oneflow
