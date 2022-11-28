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

#ifndef ONEFLOW_API_COMMON_JOB_BUILD_AND_INFER_CTX_H_
#define ONEFLOW_API_COMMON_JOB_BUILD_AND_INFER_CTX_H_

#include "oneflow/core/job/job.pb.h"
#include "oneflow/core/job/job_build_and_infer_ctx_mgr.h"

namespace oneflow {

inline Maybe<Job> GetCurrentJob() {
  auto* job_ctx_mgr = Singleton<LazyJobBuildAndInferCtxMgr>::Get();
  CHECK_NOTNULL_OR_RETURN(job_ctx_mgr);
  auto* job_ctx =
      JUST(job_ctx_mgr->FindJobBuildAndInferCtx(*JUST(job_ctx_mgr->GetCurrentJobName())));
  CHECK_NOTNULL_OR_RETURN(job_ctx);
  return job_ctx->job();
}

}  // namespace oneflow

#endif  // ONEFLOW_API_COMMON_JOB_BUILD_AND_INFER_CTX_H_
