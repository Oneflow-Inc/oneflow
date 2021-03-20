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
#include "oneflow/core/graph/compute_task_node.h"

namespace oneflow {

class WaitAndSendIdsCompTaskNode final : public CompTaskNode {
 public:
  OF_DISALLOW_COPY_AND_MOVE(WaitAndSendIdsCompTaskNode);
  WaitAndSendIdsCompTaskNode() = default;
  ~WaitAndSendIdsCompTaskNode() override = default;

  void ProduceAllRegstsAndBindEdges() override;
  void ConsumeAllRegsts() override {}
  void BuildExecGphAndRegst() override;
  bool IsMeaningLess() override { return false; }

  TaskType GetTaskType() const override { return TaskType::kWaitAndSendIds; }
  bool IsIndependent() const override { return true; }

 private:
  void InferProducedDataRegstTimeShape() override;
};

void WaitAndSendIdsCompTaskNode::ProduceAllRegstsAndBindEdges() {
  std::shared_ptr<RegstDesc> out_regst = ProduceRegst("out", false, 100, 100);
  ForEachOutDataEdge([&](TaskEdge* edge) { edge->AddRegst("out", out_regst); });
}

void WaitAndSendIdsCompTaskNode::BuildExecGphAndRegst() {
  std::shared_ptr<RegstDesc> out_regst = GetProducedRegst("out");
  ExecNode* node = mut_exec_gph().NewNode();
  node->mut_op() = op();
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

REGISTER_COMPUTE_TASK_NODE_STREAM_INDEX_GETTER(DeviceType::kCPU, TaskType::kWaitAndSendIds)
    .SetStreamIndexGetterFn([](CPUStreamIndexGenerator* generator) -> uint32_t {
      return generator->GenerateIndependentTaskStreamIndex(TaskType::kWaitAndSendIds);
    });

REGISTER_SYSTEM_OP_COMP_TASK_NODE_TYPE(OperatorConf::kWaitAndSendIdsConf,
                                       WaitAndSendIdsCompTaskNode);

}  // namespace oneflow
