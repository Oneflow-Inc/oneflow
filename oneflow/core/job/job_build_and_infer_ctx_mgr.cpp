#include "oneflow/core/job/job_build_and_infer_ctx_mgr.h"

namespace oneflow {

Maybe<void> JobBuildAndInferCtxMgr::EnterJobBuildAndInferContext(const std::string& job_name) {
  TODO();
}

Maybe<JobBuildAndInferCtx> JobBuildAndInferCtxMgr::GetCurrentJobBuildAndInferCtx() { TODO(); }

Maybe<void> JobBuildAndInferCtxMgr::LeaveCurrentJobBuildAndInferCtx() { TODO(); }

}  // namespace oneflow
