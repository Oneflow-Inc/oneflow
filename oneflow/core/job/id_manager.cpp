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

struct Int64 {
  int64_t first : kRankLimitShift;
  int64_t second : (sizeof(int64_t) * 8 - kRankLimitShift);
};
static_assert(sizeof(Int64) == sizeof(int64_t), "");

}  // namespace

IDMgr::IDMgr() {
  regst_desc_id_count_ = 0;
  mem_block_id_count_ = 0;
  chunk_id_count_ = 0;
  CHECK_LE(GlobalProcessCtx::WorldSize(), (static_cast<int64_t>(1) << kRankLimitShift));
}

int64_t IDMgr::NewRegstDescId() {
  Int64 x;
  x.first = GlobalProcessCtx::Rank();
  x.second = regst_desc_id_count_++;
  return *reinterpret_cast<int64_t*>(&x);
}

int64_t IDMgr::NewMemBlockId() {
  Int64 x;
  x.first = GlobalProcessCtx::Rank();
  x.second = mem_block_id_count_++;
  return *reinterpret_cast<int64_t*>(&x);
}

int64_t IDMgr::NewChunkId() {
  Int64 x;
  x.first = GlobalProcessCtx::Rank();
  x.second = chunk_id_count_++;
  return *reinterpret_cast<int64_t*>(&x);
}

}  // namespace oneflow
