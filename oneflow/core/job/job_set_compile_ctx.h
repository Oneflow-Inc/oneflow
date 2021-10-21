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
#ifndef ONEFLOW_CORE_JOB_SET_COMPILE_CTX_
#define ONEFLOW_CORE_JOB_SET_COMPILE_CTX_

#include "oneflow/core/job/compiler.h"
#include "oneflow/core/job/job_set_compile_ctx.pb.h"

namespace oneflow {

class JobSetCompileCtx final {
 public:
  JobSetCompileCtx() = default;
  ~JobSetCompileCtx() = default;

  PbMap<std::string, int64_t>* GetVarOpName2randomSeed() {
    return job_set_compile_ctx_proto_.mutable_var_op_name2random_seed();
  }

 private:
  JobSetCompileCtxProto job_set_compile_ctx_proto_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_JOB_SET_COMPILE_CTX_
