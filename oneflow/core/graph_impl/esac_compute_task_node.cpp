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
#include "oneflow/core/graph/esac_compute_task_node.h"
#include "oneflow/core/graph/logical_node.h"
#include "oneflow/core/common/protobuf.h"

namespace oneflow {

void EsacCompTaskNode::ConsumeAllRegsts() {
  const std::shared_ptr<const Operator> op = logical_node()->SoleOp();
  HashMap<LogicalBlobId, int64_t> lbi2ibn_id;
  FOR_RANGE(int64_t, ibn_id, 0, op->input_bns().size()) {
    CHECK(lbi2ibn_id.emplace(op->BnInOp2Lbi(GenRepeatedBn("in", ibn_id)), ibn_id).second);
  }
  ForEachInDataEdge([&](TaskEdge* edge) {
    const LogicalNode* pred = GetOnePredLogicalNodeOnEdge(edge);
    int64_t ibn_id = -1;
    for (const std::string& obn : pred->SoleOp()->output_bns()) {
      const LogicalBlobId& lbi = pred->SoleOp()->BnInOp2Lbi(obn);
      if (lbi2ibn_id.find(lbi) != lbi2ibn_id.cend()) {
        CHECK_EQ(ibn_id, -1);
        ibn_id = lbi2ibn_id.at(lbi);
      }
    }
    CHECK_NE(ibn_id, -1);
    ConsumeRegst("in_" + std::to_string(ibn_id), edge->GetSoleRegst());
  });
}

void EsacCompTaskNode::ProduceAllRegstsAndBindEdges() {
  std::shared_ptr<RegstDesc> out_regst = ProduceRegst("out", false, 1, 1);
  ForEachOutDataEdge([&](TaskEdge* edge) { edge->AddRegst("out", out_regst); });
}

void EsacCompTaskNode::BuildExecGphAndRegst() {
  ExecNode* node = mut_exec_gph().NewNode();
  std::shared_ptr<const Operator> sole_op = this->logical_node()->SoleOp();
  node->mut_op() = sole_op;
  FOR_RANGE(int64_t, ibn_id, 0, sole_op->input_bns().size()) {
    node->BindBnWithRegst(GenRepeatedBn("in", ibn_id),
                          GetSoleConsumedRegst("in_" + std::to_string(ibn_id)));
  }
  std::shared_ptr<RegstDesc> out_regst = GetProducedRegst("out");
  out_regst->AddLbi(sole_op->BnInOp2Lbi("out"));
  node->BindBnWithRegst("out", out_regst);
  node->InferBlobDescs(parallel_ctx());
}

void EsacCompTaskNode::InferProducedDataRegstTimeShape() { NaiveInferProducedDataRegstTimeShape(); }

REGISTER_TICK_TOCK_TASK_TYPE(TaskType::kEsac);

}  // namespace oneflow
