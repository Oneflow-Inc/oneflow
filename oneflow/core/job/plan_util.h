#ifndef ONEFLOW_CORE_JOB_PLAN_UTIL_H_
#define ONEFLOW_CORE_JOB_PLAN_UTIL_H_

#include "oneflow/core/job/plan.pb.h"

namespace oneflow {

struct PlanUtil {
  static RegstDescProto* GetSoleProducedDataRegst(TaskProto* task_proto);
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_JOB_PLAN_UTIL_H_
