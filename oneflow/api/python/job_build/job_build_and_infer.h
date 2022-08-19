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
#ifndef ONEFLOW_API_PYTHON_JOB_BUILD_JOB_BUILD_AND_INFER_H_
#define ONEFLOW_API_PYTHON_JOB_BUILD_JOB_BUILD_AND_INFER_H_

#include "oneflow/core/job/global_for.h"
#include "oneflow/core/common/protobuf.h"
#include "oneflow/core/framework/tensor.h"
#include "oneflow/core/framework/tensor_tuple.h"
#include "oneflow/core/framework/tensor_name_scope.h"
#include "oneflow/core/job/job_build_and_infer_ctx.h"
#include "oneflow/core/job/job_build_and_infer_ctx_mgr.h"
#include "oneflow/core/job/job.pb.h"
#include "oneflow/core/job/lazy_mode.h"
#include "oneflow/core/record/record.pb.h"

namespace oneflow {

inline Maybe<void> JobBuildAndInferCtx_Open(const std::string& job_name) {
  auto* mgr = JUST(GlobalJobBuildAndInferCtxMgr());
  return mgr->OpenJobBuildAndInferCtx(job_name);
}

inline Maybe<std::string> JobBuildAndInferCtx_GetCurrentJobName() {
  auto* mgr = JUST(GlobalJobBuildAndInferCtxMgr());
  return mgr->GetCurrentJobName();
}

inline Maybe<int64_t> JobBuildAndInferCtx_GetCurrentJobId() {
  return JUST(GetCurInferCtx())->job_id();
}

inline Maybe<void> JobBuildAndInferCtx_Close() {
  auto* mgr = JUST(GlobalJobBuildAndInferCtxMgr());
  JUST(mgr->CloseCurrentJobBuildAndInferCtx());
  return Maybe<void>::Ok();
}

inline Maybe<void> CurJobBuildAndInferCtx_SetJobConf(const std::string& job_conf_str) {
  JobConfigProto job_conf;
  CHECK_OR_RETURN(TxtString2PbMessage(job_conf_str, &job_conf)) << "job conf parse failed";
  return JUST(GetCurInferCtx())->SetJobConf(job_conf);
}

inline Maybe<void> CurJobBuildAndInferCtx_Complete() { return JUST(GetCurInferCtx())->Complete(); }

inline Maybe<void> AddTensorAsGraphLoss(const std::shared_ptr<one::Tensor>& t) {
  CHECK_OR_RETURN(t->is_lazy());
  CHECK_OR_RETURN(LazyMode::is_enabled());
  const std::string& loss_lbn = one::TensorNameScope::Global()->Lookup(t);
  CHECK_OR_RETURN("" != loss_lbn);
  return JUST(GetCurInferCtx())->AddLossLogicalBlobName(loss_lbn);
}

Maybe<void> MarkVariableGradients(const one::TensorTuple& variables,
                                  const one::TensorTuple& gradients);

Maybe<void> MarkOutputGradients(const one::TensorTuple& outputs, const one::TensorTuple& gradients);

}  // namespace oneflow

#endif  // ONEFLOW_API_PYTHON_JOB_BUILD_JOB_BUILD_AND_INFER_H_
