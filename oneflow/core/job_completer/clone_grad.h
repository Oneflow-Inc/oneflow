#ifndef ONEFLOW_CORE_AUTOGRAD_CLONE_GRAD_H_
#define ONEFLOW_CORE_AUTOGRAD_CLONE_GRAD_H_

#include "oneflow/core/autograd/autograd.h"

namespace oneflow {

void GenerateCloneGradOpIfNeed(
    const Operator& op, std::vector<OperatorConf>* op_confs,
    const HashMap<LogicalBlobId, HashMap<std::string, LogicalBlobId>>& lbi2op_name2in_diff_lbi,
    HashMap<LogicalBlobId, LogicalBlobId>* lbi2out_diff_lbi);
}

#endif  // ONEFLOW_CORE_AUTOGRAD_CLONE_GRAD_H_
