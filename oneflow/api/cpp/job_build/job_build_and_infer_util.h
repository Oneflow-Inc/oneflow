#ifndef ONEFLOW_API_CPP_JOB_BUILD_JOB_BUILD_AND_INFER_UTIL_H_
#define ONEFLOW_API_CPP_JOB_BUILD_JOB_BUILD_AND_INFER_UTIL_H_

#include <string>
#include "oneflow/core/control/global_process_ctx.h"
#include "oneflow/core/job/oneflow.h"
#include "oneflow/core/job/inter_user_job_info.pb.h"
#include "oneflow/core/job/job_build_and_infer_ctx_mgr.h"

namespace oneflow {

inline Maybe<std::string> CurJobBuildAndInferCtx_AddAndInferConsistentOp(
    const OperatorConf& op_conf) {
  auto* ctx = JUST(GetCurInferCtx());
  const auto& op_attribute = JUST(ctx->AddAndInferConsistentOp(op_conf));
  return PbMessage2TxtString(*op_attribute);
}

inline Maybe<void> CurJobBuildAndInferCtx_SetJobConf(const JobConfigProto& job_conf) {
  return JUST(GetCurInferCtx())->SetJobConf(job_conf);
}

}  // namespace oneflow

#endif  // ONEFLOW_API_CPP_JOB_BUILD_JOB_BUILD_AND_INFER_UTIL_H_