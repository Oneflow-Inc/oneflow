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
  Maybe<JobBuildAndInferCtx> FindJobBuildAndInferCtx(const std::string& job_name);
  Maybe<std::string> GetCurrentJobName();
  Maybe<void> LeaveCurrentJobBuildAndInferCtx();

 private:
  JobBuildAndInferCtxMgr() = default;
  friend class Global<JobBuildAndInferCtxMgr>;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_JOB_JOB_BUILD_AND_INFER_CXT_MGR_H_
