#ifndef ONEFLOW_CORE_JOB_INTER_JOB_MEM_SHARING_UTIL_H_
#define ONEFLOW_CORE_JOB_INTER_JOB_MEM_SHARING_UTIL_H_

#include "oneflow/core/job/job_set.pb.h"
#include "oneflow/core/job/plan.pb.h"

namespace oneflow {

struct InterJobMemSharingUtil {
  static void BindInterfaceMemBlockId(const std::vector<Job>& jobs, std::vector<Plan>* sub_plans);

  static void MergeMemBlockBetweenSubPlans(const std::vector<Job>& jobs,
                                           std::vector<Plan>* sub_plans);
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_JOB_INTER_JOB_MEM_SHARING_UTIL_H_
