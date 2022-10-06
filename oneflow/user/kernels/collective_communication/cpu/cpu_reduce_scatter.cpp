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
#include "oneflow/user/kernels/collective_communication/include/reduce_scatter.h"
#include "oneflow/user/kernels/collective_communication/cpu/cpu_collective_communication_util.h"

namespace oneflow {

namespace ccl {

namespace {

template<typename T, ReduceType reduce_type>
struct ReduceScatterImpl final {
  static Maybe<void> Call(const void* void_in, void* void_out, size_t elem_cnt,
                          Symbol<ParallelDesc> parallel_desc) {
    int64_t parallel_num = parallel_desc->parallel_num();
    if (parallel_num == 1) {
      if (void_in != void_out) { std::memcpy(void_out, void_in, elem_cnt * sizeof(T)); }
      return Maybe<void>::Ok();
    }

    const T* in = reinterpret_cast<const T*>(void_in);
    T* out = reinterpret_cast<T*>(void_out);

    BalancedSplitter bs(elem_cnt * parallel_num, parallel_num);
    const auto& opt_parallel_id = JUST(GetParallelId4CurrentProcessCtx(parallel_desc));
    CHECK_OR_RETURN(opt_parallel_id->has_value()) << kOfBugIssueUploadPrompt;
    int64_t parallel_id = JUST(*opt_parallel_id);

    auto recv_buffer = std::make_unique<T[]>(bs.At(0).size());
    const auto& rank_group = JUST(RankGroup::New(parallel_desc));

    TransportToken transport_token =
        JUST(TransportToken::NewTransportToken(kTransportTokenTypeData));
    for (int64_t i = 0, part_id = RingDecrease(parallel_id, parallel_num); i < parallel_num - 1;
         ++i, part_id = RingDecrease(part_id, parallel_num)) {
      int64_t send_part_id = part_id;
      const T* send_ptr = nullptr;
      if (i == 0) {
        send_ptr = &in[bs.At(send_part_id).begin()];
      } else {
        send_ptr = out;
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
      if (recv_size > 0) { ReduceFunctor<T, reduce_type>::Call(recv_size, out, cur_in, recv_ptr); }
    }
    return Maybe<void>::Ok();
  }
};

#define MAKE_ALL_REDUCE_ENTRY(func_name, T, reduce_type) func_name<T, reduce_type>::Call

DEFINE_STATIC_SWITCH_FUNC(Maybe<void>, ReduceScatterImpl, MAKE_ALL_REDUCE_ENTRY,  // NOLINT
                          MAKE_DATA_TYPE_CTRV_SEQ(POD_DATA_TYPE_SEQ),             // NOLINT
                          REDUCE_TYPE_CTRV_SEQ);                                  // NOLINT

#undef MAKE_ALL_REDUCE_ENTRY

}  // namespace

class CpuReduceScatter final : public ReduceScatter {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CpuReduceScatter);
  CpuReduceScatter() : datatype_(kInvalidDataType), reduce_type_(kInvalidReduceFunctorType) {}
  ~CpuReduceScatter() = default;

  void Init(DataType datatype, ReduceType reduce_type) override {
    this->datatype_ = datatype;
    this->reduce_type_ = reduce_type;
  }

  void Launch(ep::Stream* stream, const void* in, void* out, size_t elem_cnt,
              const std::shared_ptr<CommunicationContext>& communication_ctx) const override {
    const auto& cpu_communication_ctx =
        std::dynamic_pointer_cast<CpuCommunicationContext>(communication_ctx);
    CHECK(cpu_communication_ctx) << kOfBugIssueUploadPrompt;
    CHECK_JUST(SwitchReduceScatterImpl(SwitchCase(datatype_, reduce_type_), in, out, elem_cnt,
                                       cpu_communication_ctx->parallel_desc()));
  }

 private:
  DataType datatype_;
  ReduceType reduce_type_;
};

REGISTER_COLLECTIVE_COMMUNICATION(DeviceType::kCPU, ReduceScatter, CpuReduceScatter);

}  // namespace ccl

}  // namespace oneflow
