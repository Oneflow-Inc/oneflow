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
  virtual ~JobBuildAndInferCtxMgr() = default;

  Maybe<void> OpenJobBuildAndInferCtx(const std::string& job_name);
  Maybe<JobBuildAndInferCtx*> FindJobBuildAndInferCtx(const std::string& job_name);
  Maybe<std::string> GetCurrentJobName() const;
  Maybe<void> CloseCurrentJobBuildAndInferCtx();

  const JobSet& job_set() const { return job_set_; }
  std::string structure_graph() const;

 protected:
  virtual JobBuildAndInferCtx* NewJobBuildAndInferCtx(Job* job, int64_t job_id) const = 0;
  JobBuildAndInferCtxMgr() : has_cur_job_(false) {}
  virtual Maybe<void> VirtualCloseJob() = 0;
  JobSet* mut_job_set() { return &job_set_; }

  void clear_job_name2infer_ctx() { job_name2infer_ctx_.clear(); }

 private:
  JobSet job_set_;
  bool has_cur_job_;
  std::string cur_job_name_;
  HashMap<std::string, std::unique_ptr<JobBuildAndInferCtx>> job_name2infer_ctx_;
};

class LazyJobBuildAndInferCtxMgr : public JobBuildAndInferCtxMgr {
 public:
  OF_DISALLOW_COPY_AND_MOVE(LazyJobBuildAndInferCtxMgr);
  LazyJobBuildAndInferCtxMgr() : JobBuildAndInferCtxMgr() {}
  ~LazyJobBuildAndInferCtxMgr() override = default;

 private:
  friend class Global<LazyJobBuildAndInferCtxMgr>;

  Maybe<void> VirtualCloseJob() override;
  JobBuildAndInferCtx* NewJobBuildAndInferCtx(Job* job, int64_t job_id) const;
};

class EagerJobBuildAndInferCtxMgr : public JobBuildAndInferCtxMgr {
 public:
  OF_DISALLOW_COPY_AND_MOVE(EagerJobBuildAndInferCtxMgr);
  EagerJobBuildAndInferCtxMgr() : JobBuildAndInferCtxMgr() {}
  ~EagerJobBuildAndInferCtxMgr() override = default;

 private:
  friend class Global<EagerJobBuildAndInferCtxMgr>;

  Maybe<void> VirtualCloseJob() override;
  JobBuildAndInferCtx* NewJobBuildAndInferCtx(Job* job, int64_t job_id) const;
};

bool EagerExecutionEnabled();

Maybe<JobBuildAndInferCtxMgr*> GlobalJobBuildAndInferCtxMgr();
Maybe<JobBuildAndInferCtx*> GetJobBuildAndInferCtx(const std::string& job_name);
Maybe<JobBuildAndInferCtx*> GetCurInferCtx();

}  // namespace oneflow

#endif  // ONEFLOW_CORE_JOB_JOB_BUILD_AND_INFER_CXT_MGR_H_
