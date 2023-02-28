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
#include "oneflow/core/job/id_manager.h"
#include "oneflow/core/rpc/include/global_process_ctx.h"

namespace oneflow {

namespace {

constexpr static int kRankLimitShift = 16;
constexpr static int kIdLimitShift = (sizeof(int64_t) * 8 - kRankLimitShift);
static_assert(kIdLimitShift > 0, "");

int64_t AddCurrentRankOffset(int64_t x) {
  CHECK_GE(x, 0);
  CHECK_LT(x, (static_cast<int64_t>(1) << kIdLimitShift));
  return (static_cast<int64_t>(GlobalProcessCtx::Rank()) << kIdLimitShift) + x;
}

}  // namespace

IDMgr::IDMgr() {
  regst_desc_id_count_ = 0;
  mem_block_id_count_ = 0;
  chunk_id_count_ = 0;
  CHECK_LE(GlobalProcessCtx::WorldSize(), (static_cast<int64_t>(1) << kRankLimitShift));
}

int64_t IDMgr::NewRegstDescId() { return AddCurrentRankOffset(regst_desc_id_count_++); }

int64_t IDMgr::NewMemBlockId() { return AddCurrentRankOffset(mem_block_id_count_++); }

int64_t IDMgr::NewChunkId() { return AddCurrentRankOffset(chunk_id_count_++); }

}  // namespace oneflow
