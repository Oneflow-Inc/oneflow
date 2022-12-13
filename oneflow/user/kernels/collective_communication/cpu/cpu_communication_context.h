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
#ifndef ONEFLOW_USER_KERNELS_COLLECTIVE_COMMUNICATION_CPU_CPU_COMMUNICATION_CONTEXT_H_
#define ONEFLOW_USER_KERNELS_COLLECTIVE_COMMUNICATION_CPU_CPU_COMMUNICATION_CONTEXT_H_

#include "oneflow/user/kernels/collective_communication/include/communication_context.h"
#include "oneflow/core/common/symbol.h"

namespace oneflow {

class ParallelDesc;

namespace ccl {

class CpuCommunicationContext : public CommunicationContext {
 public:
  explicit CpuCommunicationContext() = default;
  ~CpuCommunicationContext() override = default;

  void Init(Symbol<ParallelDesc>) override;

  Symbol<ParallelDesc> parallel_desc() const { return parallel_desc_; }

 private:
  Symbol<ParallelDesc> parallel_desc_;
};

}  // namespace ccl

}  // namespace oneflow

#endif  // ONEFLOW_USER_KERNELS_COLLECTIVE_COMMUNICATION_CPU_CPU_COMMUNICATION_CONTEXT_H_
