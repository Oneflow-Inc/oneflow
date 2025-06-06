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
#include "oneflow/user/kernels/collective_communication/include/all_reduce.h"
#include "oneflow/user/kernels/collective_communication/cuda/cuda_communication_context.h"
#include "oneflow/core/device/nccl_util.h"

namespace oneflow {

namespace ccl {

namespace {

inline ncclRedOp_t GetNcclReduceType(ReduceType reduce_type) {
  switch (reduce_type) {
#define NCCL_REDUCE_TYPE_CASE(dtype) \
  case ReduceType::k##dtype: return ncclRedOp_t::nccl##dtype
    NCCL_REDUCE_TYPE_CASE(Sum);
    NCCL_REDUCE_TYPE_CASE(Max);
    default: PRINT_BUG_PROMPT_AND_ABORT();
  }
}

}  // namespace

class CudaAllReduce final : public AllReduce {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CudaAllReduce);
  CudaAllReduce() : nccl_datatype_(), nccl_reduce_op_() {}
  ~CudaAllReduce() = default;

  void Init(DataType datatype, ReduceType reduce_type) override {
    this->nccl_datatype_ = GetNcclDataType(datatype);
    this->nccl_reduce_op_ = GetNcclReduceType(reduce_type);
  }

  void Launch(ep::Stream* stream, const void* in, void* out, size_t elem_cnt,
              const std::shared_ptr<CommunicationContext>& communication_ctx) const override {
    const auto& cuda_communication_ctx =
        std::dynamic_pointer_cast<CudaCommunicationContext>(communication_ctx);
    CHECK(cuda_communication_ctx);
    OF_NCCL_CHECK(ncclAllReduce(in, out, elem_cnt, nccl_datatype_, nccl_reduce_op_,
                                cuda_communication_ctx->nccl_comm(),
                                stream->As<ep::CudaStream>()->cuda_stream()));
  }

  void Launch(ep::Stream* stream, const void* in, void* out, size_t elem_cnt,
              const ccl::CclComm& ccl_comm) const override {
    ncclComm_t* nccl_comm = reinterpret_cast<ncclComm_t*>(ccl_comm.getComm());
    OF_NCCL_CHECK(ncclAllReduce(in, out, elem_cnt, nccl_datatype_, nccl_reduce_op_, *nccl_comm,
                                stream->As<ep::CudaStream>()->cuda_stream()));
  }

 private:
  ncclDataType_t nccl_datatype_;
  ncclRedOp_t nccl_reduce_op_;
};

REGISTER_COLLECTIVE_COMMUNICATION(DeviceType::kCUDA, AllReduce, CudaAllReduce);

}  // namespace ccl

}  // namespace oneflow

#endif  // WITH_CUDA
