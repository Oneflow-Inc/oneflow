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
#include "oneflow/core/graph/collective_boxing_task_node.h"
#include "oneflow/core/graph/boxing/collective_boxing_util.h"

namespace oneflow {

void CollectiveBoxingGenericTaskNode::Init(int64_t machine_id, int64_t thrd_id,
                                           const OperatorConf& op_conf) {
  set_machine_id(machine_id);
  set_thrd_id(thrd_id);
  op_conf_ = op_conf;
}

void CollectiveBoxingGenericTaskNode::ProduceAllRegstsAndBindEdges() {
  if (boxing::collective::GenericOpHasOutput(
          op_conf_.collective_boxing_generic_conf().rank_desc())) {
    std::shared_ptr<RegstDesc> out_regst = ProduceRegst("out", false, 1, 1);
    this->ForEachOutDataEdge([&](TaskEdge* out_dege) { out_dege->AddRegst("out", out_regst); });
  }
}

void CollectiveBoxingGenericTaskNode::ConsumeAllRegsts() {
  this->ForEachInDataEdge(
      [&](TaskEdge* in_edge) { ConsumeRegst("in", SoleInDataEdge()->GetSoleRegst()); });
}

void CollectiveBoxingGenericTaskNode::BuildExecGphAndRegst() {
  ExecNode* node = mut_exec_gph().NewNode();
  std::shared_ptr<Operator> boxing_op = ConstructOp(op_conf_, &GlobalJobDesc());
  node->mut_op() = boxing_op;
  for (const std::string& ibn : boxing_op->input_bns()) {
    node->BindBnWithRegst(ibn, GetSoleConsumedRegst("in"));
  }
  std::shared_ptr<RegstDesc> out_regst = GetProducedRegst("out");
  for (const std::string& obn : boxing_op->output_bns()) {
    CHECK(out_regst != nullptr);
    node->BindBnWithRegst(obn, out_regst);
    out_regst->AddLbi(boxing_op->BnInOp2Lbi(obn));
  }
  node->InferBlobDescs(nullptr);
}

void CollectiveBoxingGenericTaskNode::InferProducedDataRegstTimeShape() {
  auto out_regst = GetProducedRegst("out");
  if (out_regst != nullptr) {
    out_regst->mut_data_regst_time_shape()->reset(
        new Shape({GlobalJobDesc().TotalBatchNum(), GlobalJobDesc().NumOfPiecesInBatch()}));
  }
}

}  // namespace oneflow
