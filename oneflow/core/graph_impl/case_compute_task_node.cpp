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
#include "oneflow/core/graph/case_compute_task_node.h"
#include "oneflow/core/graph/logical_node.h"
#include "oneflow/core/common/protobuf.h"

namespace oneflow {

void CaseCompTaskNode::ConsumeAllRegsts() { ConsumeRegst("in", SoleInDataEdge()->GetSoleRegst()); }

void CaseCompTaskNode::ProduceAllRegstsAndBindEdges() {
  const std::shared_ptr<const Operator> op = logical_node()->SoleOp();
  HashMap<LogicalBlobId, int64_t> lbi2obn_id;
  FOR_RANGE(int64_t, obn_id, 0, op->output_bns().size()) {
    CHECK(lbi2obn_id.emplace(op->BnInOp2Lbi(GenRepeatedBn("out", obn_id)), obn_id).second);
  }
  ForEachOutDataEdge([&](TaskEdge* edge) {
    const LogicalNode* succ = GetOneSuccLogicalNodeOnEdge(edge);
    int64_t obn_id = -1;
    for (const std::string& ibn : succ->SoleOp()->input_bns()) {
      const LogicalBlobId& lbi = succ->SoleOp()->BnInOp2Lbi(ibn);
      if (lbi2obn_id.find(lbi) != lbi2obn_id.cend()) {
        CHECK_EQ(obn_id, -1);
        obn_id = lbi2obn_id.at(lbi);
      }
    }
    CHECK_NE(obn_id, -1);
    std::string name = "out_" + std::to_string(obn_id);
    CHECK(GetProducedRegst(name) == nullptr);
    edge->AddRegst("out", ProduceRegst(name, false));
  });
}

void CaseCompTaskNode::BuildExecGphAndRegst() {
  ExecNode* node = mut_exec_gph().NewNode();
  std::shared_ptr<const Operator> sole_op = this->logical_node()->SoleOp();
  node->mut_op() = sole_op;
  node->BindBnWithRegst("in", GetSoleConsumedRegst("in"));
  FOR_RANGE(int64_t, obn_id, 0, sole_op->output_bns().size()) {
    std::string name = "out_" + std::to_string(obn_id);
    std::shared_ptr<RegstDesc> out_regst = GetProducedRegst(name);
    out_regst->AddLbi(sole_op->BnInOp2Lbi(name));
    node->BindBnWithRegst(name, out_regst);
  }
  node->InferBlobDescs(parallel_ctx());
}

void CaseCompTaskNode::InferProducedDataRegstTimeShape() { NaiveInferProducedDataRegstTimeShape(); }

REGISTER_TICK_TOCK_TASK_TYPE(TaskType::kCase);

}  // namespace oneflow
