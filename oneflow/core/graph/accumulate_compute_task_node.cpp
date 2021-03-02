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
#include "oneflow/core/graph/logical_node.h"
#include "oneflow/core/graph/accumulate_compute_task_node.h"

namespace oneflow {

void AccumulateCompTaskNode::ProduceAllRegstsAndBindEdges() {
  auto acc_regst = ProduceRegst("acc", false);
  SoleOutDataEdge()->AddRegst("acc", acc_regst);
}

void AccumulateCompTaskNode::ConsumeAllRegsts() {
  ConsumeRegst("one", SoleInDataEdge()->GetSoleRegst());
}

void AccumulateCompTaskNode::BuildExecGphAndRegst() {
  std::shared_ptr<RegstDesc> one_regst = GetSoleConsumedRegst("one");
  std::shared_ptr<RegstDesc> acc_regst = GetProducedRegst("acc");
  acc_regst->CopyBlobDescFrom(one_regst.get());
  std::shared_ptr<const Operator> op = logical_node()->SoleOp();
  ExecNode* exec_node = mut_exec_gph().NewNode();
  exec_node->mut_op() = op;
  exec_node->BindBnWithRegst(op->SoleIbn(), one_regst);
  exec_node->BindBnWithRegst(op->SoleObn(), acc_regst);
  acc_regst->ForEachLbi([acc_regst](const LogicalBlobId& lbi) {
    const BlobDesc* blob_desc = acc_regst->GetBlobDesc(lbi);
    CHECK_EQ(blob_desc->is_dynamic(), false);
    CHECK_EQ(blob_desc->is_tensor_list(), false);
  });
}

}  // namespace oneflow
