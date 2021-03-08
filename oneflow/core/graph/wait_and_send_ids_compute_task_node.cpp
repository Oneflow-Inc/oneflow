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
#include "oneflow/core/graph/wait_and_send_ids_compute_task_node.h"
#include "oneflow/core/graph/logical_node.h"

namespace oneflow {

void WaitAndSendIdsCompTaskNode::ProduceAllRegstsAndBindEdges() {
  std::shared_ptr<RegstDesc> out_regst = ProduceRegst("out", false, 100, 100);
  ForEachOutDataEdge([&](TaskEdge* edge) { edge->AddRegst("out", out_regst); });
}

void WaitAndSendIdsCompTaskNode::BuildExecGphAndRegst() {
  std::shared_ptr<RegstDesc> out_regst = GetProducedRegst("out");
  ExecNode* node = mut_exec_gph().NewNode();
  node->mut_op() = logical_node()->SoleOp();
  for (const std::string& obn : node->op()->output_bns()) {
    const LogicalBlobId& lbi = node->op()->BnInOp2Lbi(obn);
    out_regst->AddLbi(lbi);
    node->BindBnWithRegst(obn, out_regst);
  }
  node->InferBlobDescs(parallel_ctx());
}

void WaitAndSendIdsCompTaskNode::InferProducedDataRegstTimeShape() {
  std::shared_ptr<Shape> time_shape(new Shape({1, 1}));
  ForEachProducedDataRegst([time_shape](const std::string& name, RegstDesc* regst) {
    *regst->mut_data_regst_time_shape() = time_shape;
  });
}

}  // namespace oneflow
