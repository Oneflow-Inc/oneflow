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
#include "oneflow/core/ccl/ccl.h"
#include "oneflow/core/device/nccl_util.h"
#include "oneflow/core/framework/transport_util.h"
#include "oneflow/core/job/parallel_desc.h"
#include "oneflow/core/job/rank_group.h"
#include "oneflow/core/common/data_type.h"
#include "oneflow/core/common/data_type_seq.h"
#include "oneflow/core/common/balanced_splitter.h"
#include "oneflow/core/rpc/include/global_process_ctx.h"
#include "oneflow/core/thread/thread_manager.h"
#include "oneflow/core/job/eager_nccl_comm_manager.h"
#include "oneflow/core/ep/cuda/cuda_stream.h"
#include "oneflow/core/common/constant.h"

namespace oneflow {
namespace ccl {

namespace {

Maybe<void> InitBroadcastRankHeap(std::vector<int64_t>* ranks, const ParallelDesc& parallel_desc,
                                  int64_t root) {
  CHECK_EQ_OR_RETURN(parallel_desc.parallel_num(), parallel_desc.sorted_machine_ids().size());
  ranks->resize(parallel_desc.parallel_num());
  int64_t root_index = -1;
  for (int64_t parallel_id = 0; parallel_id < parallel_desc.parallel_num(); ++parallel_id) {
    int64_t machine_id = JUST(parallel_desc.MachineId4ParallelId(parallel_id));
    if (machine_id == root) { root_index = parallel_id; }
    (*ranks)[parallel_id] = machine_id;
  }
  CHECK_NE_OR_RETURN(root_index, -1);
  std::swap((*ranks)[0], (*ranks)[root_index]);
  return Maybe<void>::Ok();
}

}  // namespace

Maybe<void> CpuBroadcast(const void* in, void* out, size_t buffer_size, int64_t root,
                         Symbol<ParallelDesc> parallel_desc,
                         const TransportToken& transport_token) {
  static thread_local std::vector<int64_t> rank_heap{};
  JUST(InitBroadcastRankHeap(&rank_heap, *parallel_desc, root));
  auto Send = [&](void** buffer, std::size_t* size, std::function<void()>* Cb) -> Maybe<void> {
    *buffer = (root == GlobalProcessCtx::Rank() ? const_cast<void*>(in) : out);
    *size = buffer_size;
    *Cb = [] {};
    return Maybe<void>::Ok();
  };
  auto Recv = [&](void** buffer, std::size_t* size, std::function<void()>* Cb) -> Maybe<void> {
    *buffer = out;
    *size = buffer_size;
    *Cb = [] {};
    return Maybe<void>::Ok();
  };
  {
    NaiveAsyncTransportCtx transport_ctx(transport_token, Send, Recv);
    JUST(TransportUtil::ReceiveDataFromParentInHeap(rank_heap, transport_token, &transport_ctx));
    JUST_MSG(transport_ctx.WaitDone(), kAsymmetricCodeErrorMsg);
  }
  {
    NaiveAsyncTransportCtx transport_ctx(transport_token, Send, Recv);
    JUST(TransportUtil::SendDataToChildrenInHeap(rank_heap, transport_token, &transport_ctx));
    if (GlobalProcessCtx::Rank() == root && out != in) { std::memcpy(out, in, buffer_size); }
    JUST_MSG(transport_ctx.WaitDone(), kAsymmetricCodeErrorMsg);
  }
  return Maybe<void>::Ok();
}

Maybe<void> CpuSend(const void* in, size_t buffer_size, int64_t dst) {
  TransportToken transport_token = JUST(TransportToken::NewTransportToken(kTransportTokenTypeData));
  NaiveAsyncTransportCtx transport_ctx(
      transport_token,
      [&](void** buffer, std::size_t* size, std::function<void()>* Cb) -> Maybe<void> {
        *buffer = const_cast<void*>(in);
        *size = buffer_size;
        *Cb = [] {};
        return Maybe<void>::Ok();
      },
      [&](void** buffer, std::size_t* size, std::function<void()>* Cb) -> Maybe<void> {
        UNIMPLEMENTED_THEN_RETURN();
      });
  JUST(TransportUtil::SendDataToRank(dst, transport_token, &transport_ctx));
  JUST(transport_ctx.WaitDone());
  return Maybe<void>::Ok();
}

Maybe<void> CpuRecv(void* out, size_t buffer_size, int64_t src) {
  TransportToken transport_token = JUST(TransportToken::NewTransportToken(kTransportTokenTypeData));
  NaiveAsyncTransportCtx transport_ctx(
      transport_token,
      [&](void** buffer, std::size_t* size, std::function<void()>* Cb) -> Maybe<void> {
        UNIMPLEMENTED_THEN_RETURN();
      },
      [&](void** buffer, std::size_t* size, std::function<void()>* Cb) -> Maybe<void> {
        *buffer = out;
        *size = buffer_size;
        *Cb = [] {};
        return Maybe<void>::Ok();
      });
  JUST(TransportUtil::ReceiveDataFromRank(src, transport_token, &transport_ctx));
  JUST(transport_ctx.WaitDone());
  return Maybe<void>::Ok();
}

}  // namespace ccl
}  // namespace oneflow
