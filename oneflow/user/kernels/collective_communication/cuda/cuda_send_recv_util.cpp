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
#include "oneflow/user/kernels/collective_communication/cuda/cuda_send_recv_util.h"
#include "oneflow/core/rpc/include/global_process_ctx.h"
#include "oneflow/core/common/decorator.h"
#ifdef WITH_CUDA
#include "oneflow/core/job/eager_nccl_comm_manager.h"

namespace oneflow {

namespace ccl {

std::pair<ncclComm_t, int64_t> RawGetNcclCommAndPeerNcclRank(int64_t peer_process_id) {
  std::set<std::pair<int64_t, int64_t>> device_set;
  const int64_t& rank = GlobalProcessCtx::Rank();
  const int64_t peer_nccl_rank = (peer_process_id > rank) ? 1 : 0;
  device_set.emplace(rank, GlobalProcessCtx::LocalRank());
  device_set.emplace(peer_process_id, GlobalProcessCtx::LocalRank(peer_process_id));
  return {CHECK_NOTNULL(Singleton<EagerNcclCommMgr>::Get())->GetCommForDevice(device_set),
          peer_nccl_rank};
}

decltype(GetNcclCommAndPeerNcclRank) GetNcclCommAndPeerNcclRank =
    DECORATE(&RawGetNcclCommAndPeerNcclRank, ThreadLocal);

}  // namespace ccl

}  // namespace oneflow

#endif  // WITH_CUDA
