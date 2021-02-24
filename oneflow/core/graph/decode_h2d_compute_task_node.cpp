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
#include "oneflow/core/graph/decode_h2d_compute_task_node.h"
#include "oneflow/core/graph/logical_node.h"

namespace oneflow {

void DecodeH2DCompTaskNode::ConsumeAllRegsts() {
  ConsumeRegst("in", SoleInDataEdge()->GetSoleRegst());
}

void DecodeH2DCompTaskNode::ProduceAllRegstsAndBindEdges() {
  std::shared_ptr<RegstDesc> out_regst = ProduceRegst("out", false, 2, 2);
  ForEachOutDataEdge([&](TaskEdge* edge) { edge->AddRegst("out", out_regst); });
  ProduceRegst("tmp", false);
}

void DecodeH2DCompTaskNode::BuildExecGphAndRegst() {
  ExecNode* node = mut_exec_gph().NewNode();
  std::shared_ptr<const Operator> sole_op = this->logical_node()->SoleOp();
  node->mut_op() = sole_op;
  node->BindBnWithRegst(sole_op->SoleIbn(), GetSoleConsumedRegst("in"));
  std::shared_ptr<RegstDesc> out_regst = GetProducedRegst("out");
  out_regst->AddLbi(sole_op->BnInOp2Lbi(sole_op->SoleObn()));
  node->BindBnWithRegst(sole_op->SoleObn(), out_regst);
  node->AddBnToRegstAndBindIt(&Operator::tmp_bns, GetProducedRegst("tmp"));
  node->InferBlobDescs(parallel_ctx());
}

void DecodeH2DCompTaskNode::InferProducedDataRegstTimeShape() {
  NaiveInferProducedDataRegstTimeShape();
}

REGISTER_COMPUTE_TASK_NODE_STREAM_INDEX_GETTER(DeviceType::kGPU, TaskType::kDecodeH2D)                            \
  .SetStreamIndexGetterFn([](const CompTaskNode* comp_task_node,                                                  \
                             std::function<uint32_t(int task_type)> Counter,                                      \
                             std::function<uint32_t(const TaskNode*)> AllocateCpuStreamIndexEvenly) -> uint32_t { \
      return CudaStreamIndex::kDecodeH2D;                                                                         \
  });

REGISTER_COMPUTE_TASK_NODE_STREAM_INDEX_GETTER(DeviceType::kCPU, TaskType::kDecodeH2D)                            \
  .SetStreamIndexGetterFn([](const CompTaskNode* comp_task_node,                                                  \
                             std::function<uint32_t(int task_type)> Counter,                                      \
                             std::function<uint32_t(const TaskNode*)> AllocateCpuStreamIndexEvenly) -> uint32_t { \
    if (comp_task_node->IsIndependent()) {                                                                        \
      return StreamIndex::Independent(TaskType::kDecodeH2D, Counter);                                             \
    } else {                                                                                                      \
      return AllocateCpuStreamIndexEvenly(comp_task_node);                                                        \
    }                                                                                                             \
  });

}  // namespace oneflow
