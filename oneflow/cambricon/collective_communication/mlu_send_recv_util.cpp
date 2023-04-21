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
#include "oneflow/cambricon/collective_communication/mlu_send_recv_util.h"
#include "oneflow/core/rpc/include/global_process_ctx.h"
#include "oneflow/core/common/decorator.h"
#include "oneflow/cambricon/collective_communication/eager_cncl_comm_manager.h"

namespace oneflow {

namespace ccl {

std::pair<cnclComm_t, int64_t> RawGetCnclCommAndPeerCnclRank(int64_t peer_process_id) {
  std::set<std::pair<int64_t, int64_t>> device_set;
  const int64_t& rank = GlobalProcessCtx::Rank();
  const int64_t peer_cncl_rank = (peer_process_id > rank) ? 1 : 0;
  device_set.emplace(rank, GlobalProcessCtx::LocalRank());
  device_set.emplace(peer_process_id, GlobalProcessCtx::LocalRank(peer_process_id));
  return {CHECK_NOTNULL(Singleton<EagerCclCommMgr>::Get())
              ->As<EagerCnclCommMgr>()
              ->GetCommForDevice(device_set),
          peer_cncl_rank};
}

decltype(GetCnclCommAndPeerCnclRank) GetCnclCommAndPeerCnclRank =
    DECORATE(&RawGetCnclCommAndPeerCnclRank, ThreadLocal);

}  // namespace ccl

}  // namespace oneflow
