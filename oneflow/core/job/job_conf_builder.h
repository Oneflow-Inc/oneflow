#ifndef ONEFLOW_CORE_JOB_JOB_CONF_BUILDER_H_
#define ONEFLOW_CORE_JOB_JOB_CONF_BUILDER_H_

#include "oneflow/core/job/job_desc.h"

namespace oneflow {

class JobConfBuilder final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(JobConfBuilder);
  explicit JobConfBuilder(JobConf1* job_conf) : job_conf_(job_conf) {}
  ~JobConfBuilder() = default;

  void AddOps(const ParallelConf& parallel_conf, const std::vector<OperatorConf>& op_conf) const;

 private:
  JobConf1* job_conf_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_JOB_JOB_CONF_BUILDER_H_
