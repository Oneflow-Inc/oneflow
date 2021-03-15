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
#include "oneflow/core/control/ctrl_bootstrap.pb.h"
#include "oneflow/core/control/global_process_ctx.h"

namespace oneflow {

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
  return int64_t(Rank() / NumOfProcessPerNode());
}

int64_t GlobalProcessCtx::NumOfProcessPerNode() {
  if (Global<NumProcessPerNode>::Get() != nullptr) {
    return int64_t(Global<NumProcessPerNode>::Get()->value());
  }
  CHECK_NOTNULL(Global<ProcessCtx>::Get());
  CHECK_EQ(WorldSize() % NodeSize(), 0);
  return int64_t(WorldSize() / NodeSize());
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
