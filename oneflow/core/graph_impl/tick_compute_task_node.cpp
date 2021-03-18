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
#include "oneflow/core/graph/tick_compute_task_node.h"

namespace oneflow {

void TickCompTaskNode::ProduceAllRegstsAndBindEdges() {
  ProduceRegst("out", false, 1, 1);
  ForEachOutDataEdge([&](TaskEdge* edge) { BindEdgeWithProducedRegst(edge, "out"); });
}

void TickCompTaskNode::ConsumeAllRegsts() {
  ConsumeRegst("in");
  ForEachInDataEdge([&](TaskEdge* edge) { ConsumeRegst("in", edge->GetSoleRegst()); });
}

void TickCompTaskNode::BuildExecGphAndRegst() {
  ExecNode* node = mut_exec_gph().NewNode();
  node->mut_op() = shared_op();
  const std::list<std::shared_ptr<RegstDesc>>& in_regsts = GetConsumedRegst("in");
  for (const std::string& ibn : node->op()->input_bns()) {
    node->BindBnWithOneOfTheRegsts(ibn, in_regsts);
  }
  std::shared_ptr<RegstDesc> out_regst = GetProducedRegst("out");
  for (const std::string& obn : node->op()->output_bns()) {
    const LogicalBlobId& lbi = node->op()->BnInOp2Lbi(obn);
    out_regst->AddLbi(lbi);
    node->BindBnWithRegst(obn, out_regst);
  }
  node->InferBlobDescs(parallel_ctx());
}

REGISTER_TICK_TOCK_TASK_TYPE(TaskType::kTick);

REGISTER_COMPUTE_TASK_NODE_STREAM_INDEX_GETTER(DeviceType::kCPU, TaskType::kTick)
    .SetStreamIndexGetterFn([](CPUStreamIndexGenerator* generator) -> uint32_t {
      return generator->GenerateTickTockStreamIndex();
    });

REGISTER_SYSTEM_OP_COMP_TASK_NODE_TYPE(OperatorConf::kTickConf, TickCompTaskNode);

REGISTER_SYSTEM_OP_COMP_TASK_NODE_TYPE(OperatorConf::kSinkTickConf, TickCompTaskNode);

}  // namespace oneflow
