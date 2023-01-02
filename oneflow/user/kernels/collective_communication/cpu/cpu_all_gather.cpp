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
#include "oneflow/user/kernels/collective_communication/include/all_gather.h"
#include "oneflow/user/kernels/collective_communication/cpu/cpu_collective_communication_util.h"

namespace oneflow {

namespace ccl {

namespace {

Maybe<void> AllGatherImpl(const void* in, void* out, size_t elem_cnt, DataType dtype,
                          Symbol<ParallelDesc> parallel_desc) {
  int64_t parallel_num = parallel_desc->parallel_num();
  if (parallel_num == 1) {
    if (in != out) { std::memcpy(out, in, elem_cnt * GetSizeOfDataType(dtype)); }
    return Maybe<void>::Ok();
  }
  char* char_out = reinterpret_cast<char*>(out);
  size_t chunk_size = elem_cnt * GetSizeOfDataType(dtype);
  BalancedSplitter bs(chunk_size * parallel_num, parallel_num);
  const auto& opt_parallel_id = JUST(GetParallelId4CurrentProcessCtx(parallel_desc));
  CHECK_OR_RETURN(opt_parallel_id->has_value()) << kOfBugIssueUploadPrompt;
  const auto& rank_group = JUST(RankGroup::New(parallel_desc));
  TransportToken transport_token = JUST(TransportToken::NewTransportToken(kTransportTokenTypeData));
  int64_t parallel_id = JUST(*opt_parallel_id);
  // In-place operation will happen if in == out + parallel_id * chunk_size
  if (in != &char_out[parallel_id * chunk_size]) {
    memcpy(&char_out[parallel_id * chunk_size], in, chunk_size);
  }
  for (int64_t i = 0, part_id = parallel_id; i < parallel_num - 1;
       ++i, part_id = RingDecrease(part_id, parallel_num)) {
    int64_t send_part_id = part_id;
    const void* send_ptr = &char_out[bs.At(send_part_id).begin()];
    size_t send_size = bs.At(send_part_id).size();
    int64_t recv_part_id = RingDecrease(part_id, parallel_num);
    void* recv_ptr = &char_out[bs.At(recv_part_id).begin()];
    size_t recv_size = bs.At(recv_part_id).size();
    NaiveAsyncTransportCtx ctx(
        transport_token,
        [&](void** buffer, std::size_t* size, std::function<void()>* Cb) -> Maybe<void> {
          *buffer = const_cast<void*>(send_ptr);
          *size = send_size;
          *Cb = [] {};
          return Maybe<void>::Ok();
        },
        [&](void** buffer, std::size_t* size, std::function<void()>* Cb) -> Maybe<void> {
          *buffer = recv_ptr;
          *size = recv_size;
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
}  // namespace

class CpuAllGather final : public AllGather {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CpuAllGather);
  CpuAllGather() : datatype_(kInvalidDataType) {}
  ~CpuAllGather() = default;

  void Init(DataType datatype) override { this->datatype_ = datatype; }

  void Launch(ep::Stream* stream, const void* in, void* out, size_t elem_cnt,
              const std::shared_ptr<CommunicationContext>& communication_ctx) const override {
    const auto& cpu_communication_ctx =
        std::dynamic_pointer_cast<CpuCommunicationContext>(communication_ctx);
    CHECK(cpu_communication_ctx);
    CHECK_JUST(AllGatherImpl(in, out, elem_cnt, datatype_, cpu_communication_ctx->parallel_desc()));
  }

 private:
  DataType datatype_;
};

REGISTER_COLLECTIVE_COMMUNICATION(DeviceType::kCPU, AllGather, CpuAllGather);

}  // namespace ccl

}  // namespace oneflow
