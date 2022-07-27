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
#include "oneflow/core/ccl/include/all_reduce.h"
#include "oneflow/core/ccl/cuda/cuda_communicator.h"
#include "oneflow/core/device/nccl_util.h"

namespace oneflow {

namespace ccl {

namespace collective_communication {

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
  CudaAllReduce(DataType datatype, ReduceType reduce_type)
      : datatype_(datatype), reduce_type_(reduce_type) {}
  ~CudaAllReduce() = default;

  void Launch(ep::Stream* stream, const void* in, void* out, size_t elem_cnt,
              const std::shared_ptr<Communicator>& communicator) const override {
    const auto& cuda_communicator = std::dynamic_pointer_cast<CudaCommunicator>(communicator);
    CHECK(cuda_communicator);
    OF_NCCL_CHECK(ncclAllReduce(in, out, elem_cnt, GetNcclDataType(datatype_),
                                GetNcclReduceType(reduce_type_), cuda_communicator->nccl_comm(),
                                stream->As<ep::CudaStream>()->cuda_stream()));
  }

 private:
  DataType datatype_;
  ReduceType reduce_type_;
};

class CudaAllReduceFactory : public AllReduceFactory {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CudaAllReduceFactory);
  CudaAllReduceFactory() = default;
  ~CudaAllReduceFactory() override = default;

  std::unique_ptr<AllReduce> New(DataType datatype, ReduceType reduce_type) override {
    return std::make_unique<CudaAllReduce>(datatype, reduce_type);
  }
};

REGISTER_COLLECTIVE_COMMUNICATION_FACTORY(DeviceType::kCUDA, AllReduceFactory,
                                          CudaAllReduceFactory);

}  // namespace collective_communication

}  // namespace ccl

}  // namespace oneflow

#endif  // WITH_CUDA
