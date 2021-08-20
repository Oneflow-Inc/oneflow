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
#include "oneflow/core/common/data_type.h"
#include "oneflow/core/rpc/include/global_process_ctx.h"
#include "oneflow/core/job/eager_nccl_comm_manager.h"

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

template<>
Maybe<void> Broadcast<DeviceType::kCPU>(const void* in, void* out, size_t elem_cnt, DataType dtype,
                                        int64_t root, Symbol<ParallelDesc> parallel_desc,
                                        DeviceCtx* ctx) {
  CHECK_EQ_OR_RETURN(parallel_desc->device_type(), DeviceType::kCPU);
  CHECK_OR_RETURN(IsPODDataType(dtype));
  size_t buffer_size = elem_cnt * GetSizeOfDataType(dtype);
  TransportToken transport_token = JUST(TransportToken::NewDataTransportToken(parallel_desc));
  return CpuBroadcast(in, out, buffer_size, root, parallel_desc, transport_token);
}

Maybe<void> CpuBroadcast(const void* in, void* out, size_t buffer_size, int64_t root,
                         Symbol<ParallelDesc> parallel_desc,
                         const TransportToken& transport_token) {
  static thread_local std::vector<int64_t> rank_heap{};
  JUST(InitBroadcastRankHeap(&rank_heap, *parallel_desc, root));
  NaiveAsyncTransportCtx transport_ctx(
      transport_token,
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
  JUST(TransportUtil::ReceiveDataFromParentInHeap(rank_heap, transport_token, &transport_ctx));
  JUST(TransportUtil::WaitUntilDoneOrTimeout(transport_ctx, TransportUtil::TimeoutSeconds()));
  JUST(transport_token.TryAcquireCtrlTransportTokenLock());
  JUST(TransportUtil::SendDataToChildrenInHeap(rank_heap, transport_token, &transport_ctx));
  if (GlobalProcessCtx::Rank() == root && out != in) { std::memcpy(out, in, buffer_size); }
  JUST(TransportUtil::WaitUntilDoneOrTimeout(transport_ctx, TransportUtil::TimeoutSeconds()));
  return Maybe<void>::Ok();
}

Symbol<ParallelDesc> RawGetParallelDesc(int64_t peer_process_id, DeviceType device_type) {
  ParallelConf parallel_conf;
  parallel_conf.set_device_tag(*CHECK_JUST(DeviceTag4DeviceType(device_type)));
  parallel_conf.add_device_name("@" + std::to_string(GlobalProcessCtx::Rank()) + ":"
                                + std::to_string(GlobalProcessCtx::LocalRank()));
  parallel_conf.add_device_name("@" + std::to_string(peer_process_id) + ":"
                                + std::to_string(GlobalProcessCtx::LocalRank(peer_process_id)));
  return SymbolOf(ParallelDesc(parallel_conf));
}
auto* GetParallelDesc = DECORATE(&RawGetParallelDesc, ThreadLocal);

ncclComm_t RawGetNcclComm(int64_t peer_process_id) {
  std::set<std::pair<int64_t, int64_t>> device_set;
  device_set.emplace(GlobalProcessCtx::Rank(), GlobalProcessCtx::LocalRank());
  device_set.emplace(peer_process_id, GlobalProcessCtx::LocalRank(peer_process_id));
  return CHECK_NOTNULL(Global<EagerNcclCommMgr>::Get())->GetCommForDevice(device_set);
}
auto* GetNcclComm = DECORATE(&RawGetNcclComm, ThreadLocal);

template<>
Maybe<void> Send<DeviceType::kCPU>(const void* in, size_t elem_cnt, DataType dtype, int64_t dst,
                                   DeviceCtx* ctx) {
  CHECK_OR_RETURN(IsPODDataType(dtype));
  size_t buffer_size = elem_cnt * GetSizeOfDataType(dtype);
  TransportToken transport_token =
      JUST(TransportToken::NewDataTransportToken(GetParallelDesc(dst, DeviceType::kCPU)));
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
  JUST(TransportUtil::WaitUntilDoneOrTimeout(transport_ctx, TransportUtil::TimeoutSeconds()));
  return Maybe<void>::Ok();
}

template<>
Maybe<void> Send<DeviceType::kGPU>(const void* in, size_t elem_cnt, DataType dtype, int64_t dst,
                                   DeviceCtx* ctx) {
  CHECK_OR_RETURN(IsPODDataType(dtype));
  const auto& comm = GetNcclComm(dst);
  OF_NCCL_CHECK_OR_RETURN(
      ncclSend(in, elem_cnt, GetNcclDataType(dtype), dst, comm, ctx->cuda_stream()));
  return Maybe<void>::Ok();
}

template<>
Maybe<void> Recv<DeviceType::kCPU>(void* out, size_t elem_cnt, DataType dtype, int64_t src,
                                   DeviceCtx* ctx) {
  CHECK_OR_RETURN(IsPODDataType(dtype));
  size_t buffer_size = elem_cnt * GetSizeOfDataType(dtype);
  TransportToken transport_token =
      JUST(TransportToken::NewDataTransportToken(GetParallelDesc(src, DeviceType::kCPU)));
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
  JUST(TransportUtil::WaitUntilDoneOrTimeout(transport_ctx, TransportUtil::TimeoutSeconds()));
  return Maybe<void>::Ok();
}

template<>
Maybe<void> Recv<DeviceType::kGPU>(void* out, size_t elem_cnt, DataType dtype, int64_t src,
                                   DeviceCtx* ctx) {
  CHECK_OR_RETURN(IsPODDataType(dtype));
  const auto& comm = GetNcclComm(src);
  OF_NCCL_CHECK_OR_RETURN(
      ncclRecv(out, elem_cnt, GetNcclDataType(dtype), src, comm, ctx->cuda_stream()));
  return Maybe<void>::Ok();
}

}  // namespace ccl
}  // namespace oneflow
