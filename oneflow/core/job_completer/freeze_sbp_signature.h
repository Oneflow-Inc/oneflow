#ifndef ONEFLOW_CORE_JOB_COMPLETER_FREEZE_SBP_SIGNATURE_H_
#define ONEFLOW_CORE_JOB_COMPLETER_FREEZE_SBP_SIGNATURE_H_

#include "oneflow/core/job/job_desc.h"
#include "oneflow/core/graph/op_graph.h"

namespace oneflow {

void FreezeSbpSignature(const OpGraph& op_graph, Job* job);

}  // namespace oneflow

#endif  // ONEFLOW_CORE_JOB_COMPLETER_FREEZE_SBP_SIGNATURE_H_
