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
#include "oneflow/core/graph/distribute_concat_compute_task_node.h"
#include "oneflow/core/graph/task_graph.h"
#include "oneflow/core/operator/variable_op.h"

namespace oneflow {

bool DistributeConcatCompTaskNode::HasBackwardCompTaskNode() { return false; }

void DistributeConcatCompTaskNode::ProduceAllRegstsAndBindEdges() {
  ProduceRegst("out", true);
  ForEachOutDataEdge([&](TaskEdge* edge) { BindEdgeWithProducedRegst(edge, "out"); });
}

void DistributeConcatCompTaskNode::ConsumeAllRegsts() {
  size_t cnt = 0;
  ForEachInDataEdge([&](TaskEdge* edge) {
    cnt += 1;
    ConsumeRegst("in", edge->GetSoleRegst());
  });
  CHECK_EQ(cnt, 1);
}

bool DistributeConcatCompTaskNode::IsReadyForBuild() {
  return GetSoleConsumedRegst("in")->IsLocked();
}

void DistributeConcatCompTaskNode::BuildExecGphAndRegst() {
  BuildExecGphStructAndBindInRegst();
  BuildOutRegst();
  mut_exec_gph().TopoForEachNode([this](ExecNode* node) { node->InferBlobDescs(parallel_ctx()); });
}

void DistributeConcatCompTaskNode::BuildExecGphStructAndBindInRegst() {
  ExecNode* cur_node = mut_exec_gph().NewNode();
  cur_node->mut_op() = shared_op();
  auto in_regst = GetSoleConsumedRegst("in");
  mut_exec_gph().ForEachNode([&](ExecNode* cur_node) {
    const auto& ibn = cur_node->op()->input_bns().Get(parallel_ctx()->parallel_id());
    cur_node->BindBnWithRegst(ibn, in_regst);
    CHECK(in_regst->HasLbi(cur_node->op()->BnInOp2Lbi(ibn)));
  });
}  // namespace oneflow

void DistributeConcatCompTaskNode::BuildOutRegst() {
  std::shared_ptr<RegstDesc> out_regst = GetProducedRegst("out");
  mut_exec_gph().ForEachNode([&](ExecNode* cur_node) {
    HashSet<LogicalBlobId> found_lbis;
    for (ExecEdge* out_edge : cur_node->out_edges()) { found_lbis.insert(out_edge->lbi()); }
    for (const std::string& obn : cur_node->op()->output_bns()) {
      out_regst->AddLbi(cur_node->op()->BnInOp2Lbi(obn));
      cur_node->BindBnWithRegst(obn, out_regst);
    }
  });
  // NOTE: we can ONLY set inplace when regst has ONLY ONE blob
  auto in_regst = GetSoleConsumedRegst("in");
  if (in_regst->NumOfLbi() == 1) {
    out_regst->set_hint_inplace_consumed_regst_desc_id(in_regst->regst_desc_id());
  }
}

REGISTER_SYSTEM_OP_COMP_TASK_NODE_TYPE(OperatorConf::kDistributeConcatConf,
                                       DistributeConcatCompTaskNode);

REGISTER_SYSTEM_OP_COMP_TASK_NODE_TYPE(OperatorConf::kDistributeAddConf,
                                       DistributeConcatCompTaskNode);

}  // namespace oneflow
