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
#ifndef ONEFLOW_CAMBRICON_COLLECTIVE_COMMUNICATION_CNCL_MLU_COMMUNICATION_CONTEXT_H_
#define ONEFLOW_CAMBRICON_COLLECTIVE_COMMUNICATION_CNCL_MLU_COMMUNICATION_CONTEXT_H_

#include "oneflow/user/kernels/collective_communication/include/communication_context.h"
#include "oneflow/core/common/symbol.h"
#include "oneflow/core/job/parallel_desc.h"

#include "oneflow/cambricon/collective_communication/cncl_util.h"

namespace oneflow {

namespace ccl {

class MluCommunicationContext : public CommunicationContext {
 public:
  explicit MluCommunicationContext() = default;
  ~MluCommunicationContext() override = default;

  void Init(Symbol<ParallelDesc>) override;

  cnclComm_t cncl_comm() const { return cncl_comm_; }
  int64_t cncl_index4rank(int rank) const { return rank2cncl_index_.at(rank); }

 private:
  cnclComm_t cncl_comm_;
  HashMap<int64_t, int64_t> rank2cncl_index_;
};

}  // namespace ccl

}  // namespace oneflow

#endif  // ONEFLOW_CAMBRICON_COLLECTIVE_COMMUNICATION_CNCL_MLU_COMMUNICATION_CONTEXT_H_
