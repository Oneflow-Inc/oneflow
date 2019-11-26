#ifndef ONEFLOW_CORE_JOB_COMPLETER_CLONE_GRAD_H_
#define ONEFLOW_CORE_JOB_COMPLETER_CLONE_GRAD_H_

#include "oneflow/core/job_completer/autograd.h"

namespace oneflow {

void GenerateCloneGradOpIfNeed(const OpNode& op_node, JobBuilder* job_builder,
                               const HashMap<OpBlobArg, LogicalBlobId>& in_oba2in_diff_lbi,
                               HashMap<OpBlobArg, LogicalBlobId>* out_oba2out_diff_lbi,
                               HashMap<OpBlobArg, LogicalBlobId>* out_oba2clone_bw_add_out_lbi);
}

#endif  // ONEFLOW_CORE_JOB_COMPLETER_CLONE_GRAD_H_
