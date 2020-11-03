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
#ifndef ONEFLOW_CORE_JOB_REWRITER_SCOPE_JOB_PASS_METHOD_H_
#define ONEFLOW_CORE_JOB_REWRITER_SCOPE_JOB_PASS_METHOD_H_

#include "oneflow/core/job_rewriter/job_pass.h"

namespace oneflow {

class ScopeJobPassMethod : public JobPassMethod {
 public:
  explicit ScopeJobPassMethod(JobPassCtx* ctx) : JobPassMethod(ctx) {}
  ~ScopeJobPassMethod() override = default;

  Maybe<int64_t> MakeScopeSymbol(const std::string& job_conf, const std::string& parallel_conf,
                                 bool is_mirrored) const;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_JOB_REWRITER_SCOPE_JOB_PASS_METHOD_H_
