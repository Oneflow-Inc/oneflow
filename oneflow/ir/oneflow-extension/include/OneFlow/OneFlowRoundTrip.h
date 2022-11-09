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
#ifndef ONEFLOW_IR_ONEFLOW_EXTENSION_INCLUDE_ONEFLOW_ROUNDTRIP_H_
#define ONEFLOW_IR_ONEFLOW_EXTENSION_INCLUDE_ONEFLOW_ROUNDTRIP_H_

#include "oneflow/core/job_rewriter/job_pass.h"

namespace oneflow {

enum IRPassType : int32_t { kBeforeAD = 0, kAfterAD = 1 };

template<IRPassType ir_pass_type>
class IRRoundTrip final : public JobPass {
 public:
  IRRoundTrip() = default;
  ~IRRoundTrip() override = default;
  bool IsEnabled(const JobPassCtx& ctx) const;
  Maybe<void> Apply(Job* job, JobPassCtx* ctx) const override;
};

}  // namespace oneflow

#endif  // ONEFLOW_IR_ONEFLOW_EXTENSION_INCLUDE_ONEFLOW_ROUNDTRIP_H_
