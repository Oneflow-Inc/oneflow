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
#include "oneflow/core/framework/rpc_util.h"
#include "oneflow/core/job/parallel_desc.h"
#include "oneflow/core/common/optional.h"
#include "oneflow/core/common/data_type.h"
#include "oneflow/core/rpc/include/global_process_ctx.h"

namespace oneflow {
namespace ccl {

namespace {

Maybe<void> InitBroadcastRankHeap(std::vector<int64_t>* ranks, const ParallelDesc& parallel_desc,
                                  int64_t root) {
  CHECK_EQ_OR_RETURN(parallel_desc.parallel_num(), parallel_desc.sorted_machine_ids().size());
  ranks->resize(parallel_desc.parallel_num());
  Optional<int64_t> root_index{};
  for (int64_t parallel_id = 0; parallel_id < parallel_desc.parallel_num(); ++parallel_id) {
    int64_t machine_id = JUST(parallel_desc.MachineId4ParallelId(parallel_id));
    if (machine_id == root) { root_index = parallel_id; }
    (*ranks)[parallel_id] = machine_id;
  }
  CHECK_OR_RETURN(root_index.has_value());
  std::swap((*ranks)[0], (*ranks)[JUST(root_index.value())]);
  return Maybe<void>::Ok();
}

}  // namespace

template<>
Maybe<void> Broadcast<DeviceType::kCPU>(const void* in, void* out, size_t elem_cnt, DataType dtype,
                                        int64_t root, Symbol<ParallelDesc> parallel_desc,
                                        DeviceCtx* ctx) {
  CHECK_EQ_OR_RETURN(parallel_desc->device_type(), DeviceType::kCPU);
  static thread_local std::vector<int64_t> rank_heap{};
  JUST(InitBroadcastRankHeap(&rank_heap, *parallel_desc, root));
  RpcToken rpc_token = RpcToken::NewDataRpcToken();
  CHECK_OR_RETURN(IsPODDataType(dtype));
  size_t buffer_size = elem_cnt * GetSizeOfDataType(dtype);
  NaiveAsyncRpcCtx rpc_ctx(
      rpc_token,
      [&](void** buffer, std::size_t* size, std::function<void()>* Cb) -> Maybe<void> {
        *buffer = (root == GlobalProcessCtx::Rank() ? const_cast<void*>(in) : out);
        *size = buffer_size;
        *Cb = [] {};
        return Maybe<void>::Ok();
      },
      [&](void** buffer, std::size_t* size, std::function<void()>* Cb) -> Maybe<void> {
        *buffer = out;
        *size = buffer_size;
        *Cb = [] {};
        return Maybe<void>::Ok();
      });
  JUST(RpcUtil::ReceiveDataFromParentInHeap(rank_heap, rpc_token, &rpc_ctx));
  JUST(RpcUtil::WaitUntilDoneOrTimeout(rpc_ctx, RpcUtil::TimeoutSeconds()));
  JUST(RpcUtil::SendDataToChildrenInHeap(rank_heap, rpc_token, &rpc_ctx));
  if (GlobalProcessCtx::Rank() == root && out != in) { std::memcpy(out, in, buffer_size); }
  JUST(RpcUtil::WaitUntilDoneOrTimeout(rpc_ctx, RpcUtil::TimeoutSeconds()));
  return Maybe<void>::Ok();
}

}  // namespace ccl
}  // namespace oneflow
