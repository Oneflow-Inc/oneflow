#ifndef ONEFLOW_API_PYTHON_SESSION_SESSION_H_
#define ONEFLOW_API_PYTHON_SESSION_SESSION_H_

#include <string>
#include <google/protobuf/text_format.h>
#include "oneflow/core/common/util.h"
#include "oneflow/core/job/job_build_and_infer_ctx_mgr.h"

namespace oneflow {

inline Maybe<std::string> GetSerializedCurrentJob() {
  auto* job_ctx_mgr = Singleton<LazyJobBuildAndInferCtxMgr>::Get();
  CHECK_NOTNULL_OR_RETURN(job_ctx_mgr);
  auto* job_ctx =
      JUST(job_ctx_mgr->FindJobBuildAndInferCtx(*JUST(job_ctx_mgr->GetCurrentJobName())));
  CHECK_NOTNULL_OR_RETURN(job_ctx);
  return job_ctx->job().SerializeAsString();
}

inline Maybe<std::string> GetFunctionConfigDef() {
  std::string ret;
  google::protobuf::TextFormat::PrintToString(GlobalFunctionConfigDef(), &ret);
  return ret;
}

inline Maybe<std::string> GetScopeConfigDef() {
  std::string ret;
  google::protobuf::TextFormat::PrintToString(GlobalScopeConfigDef(), &ret);
  return ret;
}

}  // namespace oneflow

#endif  // ONEFLOW_API_PYTHON_SESSION_SESSION_H_