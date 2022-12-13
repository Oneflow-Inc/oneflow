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
#include "oneflow/core/common/data_type.h"
#include "oneflow/core/job/rank_group.h"
#include "oneflow/core/framework/transport_util.h"
#include "oneflow/user/kernels/collective_communication/cpu/cpu_communication_context.h"
#include "oneflow/user/kernels/collective_communication/include/all_reduce.h"
#include "oneflow/user/kernels/collective_communication/cpu/cpu_collective_communication_util.h"

namespace oneflow {

namespace ccl {

namespace {

template<typename T, ReduceType reduce_type>
struct AllReduceImpl final {
  static Maybe<void> Call(const void* void_in, void* void_out, size_t elem_cnt,
                          Symbol<ParallelDesc> parallel_desc) {
    int64_t parallel_num = parallel_desc->parallel_num();
    if (parallel_num == 1) {
      if (void_in != void_out) { std::memcpy(void_out, void_in, elem_cnt * sizeof(T)); }
      return Maybe<void>::Ok();
    }
    const T* in = reinterpret_cast<const T*>(void_in);
    T* out = reinterpret_cast<T*>(void_out);
    BalancedSplitter bs(elem_cnt, parallel_num);
    auto recv_buffer = std::make_unique<T[]>(bs.At(0).size());
    Optional<int64_t> parallel_id;
    JUST(GetTensorDevice4CurrentProcessCtx(parallel_desc, &parallel_id));
    const auto& rank_group = JUST(RankGroup::New(parallel_desc));
    TransportToken transport_token =
        JUST(TransportToken::NewTransportToken(kTransportTokenTypeData));
    for (int64_t i = 0, part_id = JUST(parallel_id); i < parallel_num - 1;
         ++i, part_id = RingDecrease(part_id, parallel_num)) {
      int64_t send_part_id = part_id;
      const T* send_ptr = nullptr;
      if (i == 0) {
        send_ptr = &in[bs.At(send_part_id).begin()];
      } else {
        send_ptr = &out[bs.At(send_part_id).begin()];
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
      T* cur_out = &out[bs.At(recv_part_id).begin()];
      if (recv_size > 0) {
        ReduceFunctor<T, reduce_type>::Call(recv_size, cur_out, cur_in, recv_ptr);
      }
    }
    for (int64_t i = 0, part_id = RingIncrease(JUST(parallel_id), parallel_num);
         i < parallel_num - 1; ++i, part_id = RingDecrease(part_id, parallel_num)) {
      int64_t send_part_id = part_id;
      const T* send_ptr = &out[bs.At(send_part_id).begin()];
      size_t send_size = bs.At(send_part_id).size();
      int64_t recv_part_id = RingDecrease(part_id, parallel_num);
      T* recv_ptr = &out[bs.At(recv_part_id).begin()];
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
    }
    return Maybe<void>::Ok();
  }
};

#define MAKE_ALL_REDUCE_ENTRY(func_name, T, reduce_type) func_name<T, reduce_type>::Call

DEFINE_STATIC_SWITCH_FUNC(Maybe<void>, AllReduceImpl, MAKE_ALL_REDUCE_ENTRY,  // NOLINT
                          MAKE_DATA_TYPE_CTRV_SEQ(POD_DATA_TYPE_SEQ),         // NOLINT
                          REDUCE_TYPE_CTRV_SEQ);                              // NOLINT

#undef MAKE_ALL_REDUCE_ENTRY

}  // namespace

class CpuAllReduce final : public AllReduce {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CpuAllReduce);
  CpuAllReduce() : datatype_(kInvalidDataType), reduce_type_(kInvalidReduceFunctorType) {}
  ~CpuAllReduce() = default;

  void Init(DataType datatype, ReduceType reduce_type) override {
    this->datatype_ = datatype;
    this->reduce_type_ = reduce_type;
  }

  void Launch(ep::Stream* stream, const void* in, void* out, size_t elem_cnt,
              const std::shared_ptr<CommunicationContext>& communication_ctx) const override {
    const auto& cpu_communication_ctx =
        std::dynamic_pointer_cast<CpuCommunicationContext>(communication_ctx);
    CHECK(cpu_communication_ctx) << kOfBugIssueUploadPrompt;
    CHECK_JUST(SwitchAllReduceImpl(SwitchCase(datatype_, reduce_type_), in, out, elem_cnt,
                                   cpu_communication_ctx->parallel_desc()));
  }

 private:
  DataType datatype_;
  ReduceType reduce_type_;
};

REGISTER_COLLECTIVE_COMMUNICATION(DeviceType::kCPU, AllReduce, CpuAllReduce);

}  // namespace ccl

}  // namespace oneflow
