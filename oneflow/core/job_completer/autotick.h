#ifndef ONEFLOW_CORE_JOB_COMPLETER_AUTOTICK_H_
#define ONEFLOW_CORE_JOB_COMPLETER_AUTOTICK_H_

#include "oneflow/core/job/job_desc.h"
#include "oneflow/core/operator/operator.h"
#include "oneflow/core/graph/op_graph.h"

namespace oneflow {

void AutoTick(const OpGraph& op_graph, Job* job);
}

#endif  // ONEFLOW_CORE_JOB_COMPLETER_AUTOTICK_H_
