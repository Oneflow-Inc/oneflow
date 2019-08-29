#ifndef ONEFLOW_CORE_JOB_JOB_BUILD_AND_INFER_CXT_MGR_H_
#define ONEFLOW_CORE_JOB_JOB_BUILD_AND_INFER_CXT_MGR_H_

#include "oneflow/core/common/util.h"
#include "oneflow/core/common/maybe.h"
#include "oneflow/core/job/job.pb.h"
#include "oneflow/core/job/job_build_and_infer_ctx.h"

namespace oneflow {

class JobBuildAndInferCtxMgr {
 public:
  OF_DISALLOW_COPY_AND_MOVE(JobBuildAndInferCtxMgr);
  ~JobBuildAndInferCtxMgr() = default;

  Maybe<void> EnterJobBuildAndInferContext(const std::string& job_name);
  Maybe<JobBuildAndInferCtx> GetCurrentJobBuildAndInferCtx();
  Maybe<void> LeaveCurrentJobBuildAndInferCtx();

 private:
  JobBuildAndInferCtxMgr() : cur_infer_ctx_(nullptr) {}
  friend class Global<JobBuildAndInferCtxMgr>;

  HashMap<std::string, std::shared_ptr<JobBuildAndInferCtx>> job_name2infer_ctx_;
  std::shared_ptr<JobBuildAndInferCtx> cur_infer_ctx_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_JOB_JOB_BUILD_AND_INFER_CXT_MGR_H_
