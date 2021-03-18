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
#include "oneflow/core/graph/acc_tick_compute_task_node.h"
#include "oneflow/core/operator/acc_tick_op.h"

namespace oneflow {

void AccTickCompTaskNode::ProduceAllRegstsAndBindEdges() {
  SoleOutDataEdge()->AddRegst("out", ProduceRegst("out", false));
}

void AccTickCompTaskNode::ConsumeAllRegsts() {
  ConsumeRegst("in", SoleInDataEdge()->GetSoleRegst());
}

void AccTickCompTaskNode::BuildExecGphAndRegst() {
  std::shared_ptr<RegstDesc> in_regst = GetSoleConsumedRegst("in");
  std::shared_ptr<RegstDesc> out_regst = GetProducedRegst("out");
  std::shared_ptr<const Operator> op = shared_op();
  ExecNode* exec_node = mut_exec_gph().NewNode();
  exec_node->mut_op() = op;
  exec_node->BindBnWithRegst(op->SoleIbn(), in_regst);
  out_regst->AddLbi(op->BnInOp2Lbi(op->SoleObn()));
  exec_node->BindBnWithRegst(op->SoleObn(), out_regst);
  exec_node->InferBlobDescs(parallel_ctx());
}

REGISTER_SYSTEM_OP_COMP_TASK_NODE_TYPE(OperatorConf::kAccTickConf, AccTickCompTaskNode);

}  // namespace oneflow
