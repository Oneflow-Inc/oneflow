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
#include "oneflow/core/ccl/ccl.h"
#include "oneflow/core/job/rank_group.h"
#include "oneflow/core/framework/transport_util.h"
#include "oneflow/user/kernels/collective_communication/cpu/cpu_communication_context.h"
#include "oneflow/user/kernels/collective_communication/include/broadcast.h"

namespace oneflow {

namespace ccl {

// Use CpuBroadcastImpl to avoid name conflict
class CpuBroadcastImpl final : public Broadcast {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CpuBroadcastImpl);
  CpuBroadcastImpl() : size_of_dtype_(0) {}
  ~CpuBroadcastImpl() = default;

  void Init(DataType datatype) override {
    CHECK(IsTriviallyCopyableDataType(datatype));
    this->size_of_dtype_ = GetSizeOfDataType(datatype);
  }

  void Launch(ep::Stream* stream, const void* in, void* out, size_t elem_cnt, int64_t root,
              const std::shared_ptr<CommunicationContext>& communication_ctx) const override {
    const auto& cpu_communication_ctx =
        std::dynamic_pointer_cast<CpuCommunicationContext>(communication_ctx);
    CHECK(cpu_communication_ctx);
    size_t buffer_size = elem_cnt * size_of_dtype_;
    const auto& transport_token =
        CHECK_JUST(TransportToken::NewTransportToken(kTransportTokenTypeData));
    CHECK_JUST(CpuBroadcast(in, out, buffer_size, root, cpu_communication_ctx->parallel_desc(),
                            transport_token));
  }

 private:
  size_t size_of_dtype_;
};

REGISTER_COLLECTIVE_COMMUNICATION(DeviceType::kCPU, Broadcast, CpuBroadcastImpl);

}  // namespace ccl

}  // namespace oneflow
