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

int64_t RingDecrease(int64_t n, int64_t size) { return (n - 1 + size) % size; }

int64_t RingIncrease(int64_t n, int64_t size) { return (n + 1 + size) % size; }

template<typename T>
void VecAdd(size_t size, T* out, const T* in0, const T* in1) {
  size_t thread_num = Singleton<ThreadPool>::Get()->thread_num();
  BalancedSplitter bs(size, thread_num);
  MultiThreadLoop(thread_num, [&](size_t thread_idx) {
    size_t end = bs.At(thread_idx).end();
    for (size_t i = bs.At(thread_idx).begin(); i < end; ++i) { out[i] = in0[i] + in1[i]; }
  });
}

}  // namespace

template<>
Maybe<void> Broadcast<DeviceType::kCPU>(const void* in, void* out, size_t elem_cnt, DataType dtype,
                                        int64_t root, Symbol<ParallelDesc> parallel_desc,
                                        ep::Stream* stream) {
  CHECK_EQ_OR_RETURN(parallel_desc->device_type(), DeviceType::kCPU);
  CHECK_OR_RETURN(IsPODDataType(dtype));
  size_t buffer_size = elem_cnt * GetSizeOfDataType(dtype);
  const auto& transport_token = JUST(TransportToken::NewTransportToken(kTransportTokenTypeData));
  return CpuBroadcast(in, out, buffer_size, root, parallel_desc, transport_token);
}

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

template<typename T, ReduceType reduce_type>
struct DtypeReduce;

template<typename T>
struct DtypeReduce<T, kSum> {
  static Maybe<void> Call(const void* void_in, void* void_out, size_t elem_cnt, int64_t root,
                          Symbol<ParallelDesc> parallel_desc) {
    const T* in = reinterpret_cast<const T*>(void_in);
    T* out = reinterpret_cast<T*>(void_out);

    int64_t parallel_num = parallel_desc->parallel_num();
    BalancedSplitter bs(elem_cnt, parallel_num);

    size_t size = root == GlobalProcessCtx::Rank() && void_in != void_out ? 0 : bs.At(0).size();
    T* tmp_out = nullptr;
    // void_out is only used on rank root and ignored for other ranks.
    auto tmp_out_buffer = std::make_unique<T[]>(size);
    int64_t parallel_id_of_root =
        JUST(parallel_desc->ParallelId4MachineDeviceId(root, GlobalProcessCtx::LocalRank(root)));
    if (root == GlobalProcessCtx::Rank() && void_in != void_out) {
      tmp_out = &reinterpret_cast<T*>(void_out)[bs.At(parallel_id_of_root).begin()];
    } else {
      tmp_out = tmp_out_buffer.get();
    }

    auto recv_buffer = std::make_unique<T[]>(bs.At(0).size());
    Optional<int64_t> parallel_id;
    JUST(GetTensorDevice4CurrentProcessCtx(parallel_desc, &parallel_id));
    const auto& rank_group = JUST(RankGroup::New(parallel_desc));
    TransportToken transport_token =
        JUST(TransportToken::NewTransportToken(kTransportTokenTypeData));
    for (int64_t i = 0, part_id = RingDecrease(JUST(parallel_id), parallel_num);
         i < parallel_num - 1; ++i, part_id = RingDecrease(part_id, parallel_num)) {
      int64_t send_part_id = part_id;
      const T* send_ptr = nullptr;
      if (i == 0) {
        send_ptr = &in[bs.At(send_part_id).begin()];
      } else {
        send_ptr = tmp_out;
      }
      size_t send_size = bs.At(send_part_id).size();
      int64_t recv_part_id = RingDecrease(part_id, parallel_num);
      T* recv_ptr = recv_buffer.get();
      size_t recv_size = bs.At(recv_part_id).size();
      NaiveAsyncTransportCtx ctx(
          transport_token,
          [&](void** buffer, std::size_t* size, std::function<void()>* Cb) -> Maybe<void> {
            *buffer = const_cast<T*>(send_ptr);
            *size = send_size * sizeof(T);
            *Cb = [] {};
            return Maybe<void>::Ok();
          },
          [&](void** buffer, std::size_t* size, std::function<void()>* Cb) -> Maybe<void> {
            *buffer = recv_ptr;
            *size = recv_size * sizeof(T);
            *Cb = [] {};
            return Maybe<void>::Ok();
          });
      if (send_size > 0) {
        JUST(TransportUtil::SendToNextRankInRing(rank_group, transport_token, &ctx));
      }
      if (recv_size > 0) {
        JUST(TransportUtil::ReceiveFromPrevRankInRing(rank_group, transport_token, &ctx));
      }
      JUST(ctx.WaitDone());
      const T* cur_in = &in[bs.At(recv_part_id).begin()];
      if (recv_size > 0) { VecAdd(recv_size, tmp_out, cur_in, recv_ptr); }
    }

    if (root == GlobalProcessCtx::Rank() && void_in == void_out) {
      memcpy(&out[bs.At(parallel_id_of_root).begin()], tmp_out,
             bs.At(parallel_id_of_root).size() * sizeof(T));
    }

    for (int64_t i = 0, part_id = RingIncrease(parallel_id_of_root, parallel_num);
         i < parallel_num - 1; ++i, part_id = RingIncrease(part_id, parallel_num)) {
      int64_t send_part_id = part_id;
      int64_t src_rank = JUST(parallel_desc->MachineId4ParallelId(send_part_id));
      const T* send_ptr = tmp_out;
      size_t send_size = bs.At(send_part_id).size();
      int64_t recv_part_id = part_id;
      T* recv_ptr = &out[bs.At(recv_part_id).begin()];
      size_t recv_size = bs.At(recv_part_id).size();

      if (send_size > 0 && src_rank == GlobalProcessCtx::Rank()) {
        NaiveAsyncTransportCtx ctx(
            transport_token,
            [&](void** buffer, std::size_t* size, std::function<void()>* Cb) -> Maybe<void> {
              *buffer = const_cast<T*>(send_ptr);
              *size = send_size * sizeof(T);
              *Cb = [] {};
              return Maybe<void>::Ok();
            },
            [&](void** buffer, std::size_t* size, std::function<void()>* Cb) -> Maybe<void> {
              UNIMPLEMENTED_THEN_RETURN();
            });
        JUST(TransportUtil::SendDataToRank(root, transport_token, &ctx));
        JUST(ctx.WaitDone());
      }
      if (recv_size > 0 && root == GlobalProcessCtx::Rank()) {
        NaiveAsyncTransportCtx ctx(
            transport_token,
            [&](void** buffer, std::size_t* size, std::function<void()>* Cb) -> Maybe<void> {
              UNIMPLEMENTED_THEN_RETURN();
            },
            [&](void** buffer, std::size_t* size, std::function<void()>* Cb) -> Maybe<void> {
              *buffer = recv_ptr;
              *size = recv_size * sizeof(T);
              *Cb = [] {};
              return Maybe<void>::Ok();
            });
        JUST(TransportUtil::ReceiveDataFromRank(src_rank, transport_token, &ctx));
        JUST(ctx.WaitDone());
      }
    }
    return Maybe<void>::Ok();
  }
};

#define MAKE_REDUCE_ENTRY(func_name, T, reduce_type) func_name<T, reduce_type>::Call

DEFINE_STATIC_SWITCH_FUNC(Maybe<void>, DtypeReduce, MAKE_REDUCE_ENTRY,
                          MAKE_DATA_TYPE_CTRV_SEQ(POD_DATA_TYPE_SEQ), CCL_REDUCE_TYPE_CTRV_SEQ);

#undef MAKE_REDUCE_ENTRY

template<>
Maybe<void> Reduce<DeviceType::kCPU>(const void* in, void* out, size_t elem_cnt, DataType dtype,
                                     ReduceType reduce_type, int64_t root,
                                     Symbol<ParallelDesc> parallel_desc, ep::Stream* stream) {
  return SwitchDtypeReduce(SwitchCase(dtype, reduce_type), in, out, elem_cnt, root, parallel_desc);
}

#ifdef WITH_CUDA
std::pair<ncclComm_t, int64_t> RawGetNcclCommAndPeerNcclRank(int64_t peer_process_id) {
  std::set<std::pair<int64_t, int64_t>> device_set;
  const int64_t& rank = GlobalProcessCtx::Rank();
  const int64_t peer_nccl_rank = (peer_process_id > rank) ? 1 : 0;
  device_set.emplace(rank, GlobalProcessCtx::LocalRank());
  device_set.emplace(peer_process_id, GlobalProcessCtx::LocalRank(peer_process_id));
  return {CHECK_NOTNULL(Singleton<EagerNcclCommMgr>::Get())->GetCommForDevice(device_set),
          peer_nccl_rank};
}
auto* GetNcclCommAndPeerNcclRank = DECORATE(&RawGetNcclCommAndPeerNcclRank, ThreadLocal);
#endif

template<>
Maybe<void> Send<DeviceType::kCPU>(const void* in, size_t elem_cnt, DataType dtype, int64_t dst,
                                   ep::Stream* stream) {
  CHECK_OR_RETURN(IsPODDataType(dtype));
  size_t buffer_size = elem_cnt * GetSizeOfDataType(dtype);
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

#ifdef WITH_CUDA
template<>
Maybe<void> Send<DeviceType::kCUDA>(const void* in, size_t elem_cnt, DataType dtype, int64_t dst,
                                    ep::Stream* stream) {
#if NCCL_VERSION_CODE >= 2700
  CHECK_OR_RETURN(IsPODDataType(dtype));
  const auto& comm_and_peer_rank = GetNcclCommAndPeerNcclRank(dst);
  OF_NCCL_CHECK_OR_RETURN(ncclSend(in, elem_cnt, GetNcclDataType(dtype), comm_and_peer_rank.second,
                                   comm_and_peer_rank.first,
                                   stream->As<ep::CudaStream>()->cuda_stream()));
  return Maybe<void>::Ok();
#else
  UNIMPLEMENTED_THEN_RETURN() << "GPU send is only supported when nccl version >= 2.7"
#endif
}
#endif

template<>
Maybe<void> Recv<DeviceType::kCPU>(void* out, size_t elem_cnt, DataType dtype, int64_t src,
                                   ep::Stream* stream) {
  CHECK_OR_RETURN(IsPODDataType(dtype));
  size_t buffer_size = elem_cnt * GetSizeOfDataType(dtype);
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

#ifdef WITH_CUDA
template<>
Maybe<void> Recv<DeviceType::kCUDA>(void* out, size_t elem_cnt, DataType dtype, int64_t src,
                                    ep::Stream* stream) {
#if NCCL_VERSION_CODE >= 2700
  CHECK_OR_RETURN(IsPODDataType(dtype));
  const auto& comm_and_peer_rank = GetNcclCommAndPeerNcclRank(src);
  OF_NCCL_CHECK_OR_RETURN(ncclRecv(out, elem_cnt, GetNcclDataType(dtype), comm_and_peer_rank.second,
                                   comm_and_peer_rank.first,
                                   stream->As<ep::CudaStream>()->cuda_stream()));
  return Maybe<void>::Ok();
#else
  UNIMPLEMENTED_THEN_RETURN() << "GPU recv is only supported when nccl version >= 2.7"
#endif
}
#endif

}  // namespace ccl
}  // namespace oneflow
