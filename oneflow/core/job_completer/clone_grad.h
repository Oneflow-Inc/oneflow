#ifndef ONEFLOW_CORE_JOB_COMPLETER_CLONE_GRAD_H_
#define ONEFLOW_CORE_JOB_COMPLETER_CLONE_GRAD_H_

#include "oneflow/core/job_completer/autograd.h"

namespace oneflow {

void GenerateCloneGradOpIfNeed(
    const OpNode& op_node, std::vector<OperatorConf>* op_confs,
    const HashMap<std::string, HashMap<std::string, LogicalBlobId>>& op_name2ibn2in_diff_lbi,
    HashMap<LogicalBlobId, LogicalBlobId>* lbi2out_diff_lbi);
}

#endif  // ONEFLOW_CORE_JOB_COMPLETER_CLONE_GRAD_H_
