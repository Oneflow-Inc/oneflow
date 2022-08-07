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
#ifndef ONEFLOW_USER_KERNELS_COLLECTIVE_COMMUNICATION_CUDA_CUDA_COMMUNICATION_CONTEXT_H_
#define ONEFLOW_USER_KERNELS_COLLECTIVE_COMMUNICATION_CUDA_CUDA_COMMUNICATION_CONTEXT_H_

#include "oneflow/user/kernels/collective_communication/include/communication_context.h"
#include "oneflow/core/common/symbol.h"
#include "oneflow/core/job/parallel_desc.h"

#ifdef WITH_CUDA

#include "oneflow/core/device/cuda_util.h"

namespace oneflow {

namespace ccl {

class CudaCommunicationContext : public CommunicationContext {
 public:
  explicit CudaCommunicationContext() = default;
  ~CudaCommunicationContext() override = default;

  void Init(Symbol<ParallelDesc>) override;

  ncclComm_t nccl_comm() const { return nccl_comm_; }
  int64_t nccl_index4rank(int rank) const { return rank2nccl_index_.at(rank); }

 private:
  ncclComm_t nccl_comm_;
  HashMap<int64_t, int64_t> rank2nccl_index_;
};

}  // namespace ccl

}  // namespace oneflow

#endif  // WITH_CUDA

#endif  // ONEFLOW_USER_KERNELS_COLLECTIVE_COMMUNICATION_CUDA_CUDA_COMMUNICATION_CONTEXT_H_
