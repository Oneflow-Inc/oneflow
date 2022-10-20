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
#ifdef WITH_CUDA
#include "oneflow/user/kernels/collective_communication/include/all_gather.h"
#include "oneflow/user/kernels/collective_communication/cuda/cuda_communication_context.h"
#include "oneflow/core/device/nccl_util.h"

namespace oneflow {

namespace ccl {

class CudaAllGather final : public AllGather {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CudaAllGather);
  CudaAllGather() : nccl_datatype_() {}
  ~CudaAllGather() = default;

  void Init(DataType datatype) override { this->nccl_datatype_ = GetNcclDataType(datatype); }

  void Launch(ep::Stream* stream, const void* in, void* out, size_t elem_cnt,
              const std::shared_ptr<CommunicationContext>& communication_ctx) const override {
    const auto& cuda_communication_ctx =
        std::dynamic_pointer_cast<CudaCommunicationContext>(communication_ctx);
    CHECK(cuda_communication_ctx) << kOfBugIssueUploadPrompt;
    OF_NCCL_CHECK(ncclAllGather(in, out, elem_cnt, nccl_datatype_,
                                cuda_communication_ctx->nccl_comm(),
                                stream->As<ep::CudaStream>()->cuda_stream()));
  }

 private:
  ncclDataType_t nccl_datatype_;
};

REGISTER_COLLECTIVE_COMMUNICATION(DeviceType::kCUDA, AllGather, CudaAllGather);

}  // namespace ccl

}  // namespace oneflow

#endif  // WITH_CUDA
