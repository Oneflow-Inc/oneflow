#ifndef ONEFLOW_CORE_JOB_JOB_BUILD_AND_INFER_CXT_MGR_H_
#define ONEFLOW_CORE_JOB_JOB_BUILD_AND_INFER_CXT_MGR_H_

#include "oneflow/core/common/util.h"
#include "oneflow/core/common/maybe.h"
#include "oneflow/core/job/job.pb.h"
#include "oneflow/core/job/job_set.pb.h"
#include "oneflow/core/job/job_build_and_infer_ctx.h"
#include "oneflow/core/job/lbi_diff_watcher_info.pb.h"

namespace oneflow {

class JobBuildAndInferCtxMgr {
 public:
  OF_DISALLOW_COPY_AND_MOVE(JobBuildAndInferCtxMgr);
  ~JobBuildAndInferCtxMgr() = default;

  Maybe<void> OpenJobBuildAndInferCtx(const std::string& job_name);
  Maybe<JobBuildAndInferCtx*> FindJobBuildAndInferCtx(const std::string& job_name);
  Maybe<std::string> GetCurrentJobName() const;
  void CloseCurrentJobBuildAndInferCtx();
  Maybe<void> AddLbiAndDiffWatcherUuidPair(const LbiAndDiffWatcherUuidPair& lbi_uuid_pair) const;

  const JobSet& job_set() const { return job_set_; }

 private:
  JobBuildAndInferCtxMgr() : has_cur_job_(false) {}
  friend class Global<JobBuildAndInferCtxMgr>;

  JobSet job_set_;
  bool has_cur_job_;
  std::string cur_job_name_;
  HashMap<std::string, std::unique_ptr<JobBuildAndInferCtx>> job_name2infer_ctx_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_JOB_JOB_BUILD_AND_INFER_CXT_MGR_H_
