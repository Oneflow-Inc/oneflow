#ifndef ONEFLOW_CUSTOMIZED_CUSTOMIZED_PLAN_TO_PHYSICAL_GRAPH_H_
#define ONEFLOW_CUSTOMIZED_CUSTOMIZED_PLAN_TO_PHYSICAL_GRAPH_H_

#include "oneflow/core/job/plan.pb.h"

namespace oneflow {

void PlanToPhysicalGraphFile(const Plan& plan);

}  // namespace oneflow

#endif  // ONEFLOW_CUSTOMIZED_UTILS_PLAN_TO_PHYSICAL_GRAPH_H_
