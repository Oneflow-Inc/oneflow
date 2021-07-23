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

class ForeignIOCompTaskNode : public CompTaskNode {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ForeignIOCompTaskNode);
  ForeignIOCompTaskNode() = default;
  virtual ~ForeignIOCompTaskNode() override = default;

  void ProduceAllRegstsAndBindEdges() override;
  void ConsumeAllRegsts() override;
  void BuildExecGphAndRegst() override;
  bool IsMeaningLess() override { return false; }

  bool IsIndependent() const override { return true; }

 private:
  void InferProducedDataRegstTimeShape() override;
};

void ForeignIOCompTaskNode::ProduceAllRegstsAndBindEdges() {
  std::shared_ptr<RegstDesc> out_regst = ProduceRegst("out", false, 1, 1);
  ForEachOutDataEdge([&](TaskEdge* edge) { edge->AddRegst("out", out_regst); });
}

void ForeignIOCompTaskNode::ConsumeAllRegsts() {
  ConsumeRegst("in");
  ForEachInDataEdge([&](TaskEdge* edge) { ConsumeRegst("in", edge->GetSoleRegst()); });
}

void ForeignIOCompTaskNode::BuildExecGphAndRegst() {
  ExecNode* node = mut_exec_gph().NewNode();
  node->mut_op() = this->op();
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

void ForeignIOCompTaskNode::InferProducedDataRegstTimeShape() {
  auto time_shape = (*in_edges().begin())->src_node()->GetFastestInputOutputTimeShape();
  for (TaskEdge* edge : in_edges()) {
    CHECK(time_shape->elem_cnt() == edge->src_node()->GetFastestInputOutputTimeShape()->elem_cnt());
  }
  ForEachProducedDataRegst([time_shape](const std::string& name, RegstDesc* regst) {
    *regst->mut_data_regst_time_shape() = time_shape;
  });
}

class ForeignInputCompTaskNode final : public ForeignIOCompTaskNode {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ForeignInputCompTaskNode);
  ForeignInputCompTaskNode() = default;
  ~ForeignInputCompTaskNode() override = default;

  TaskType GetTaskType() const override { return TaskType::kForeignInput; }
};

class ForeignOutputCompTaskNode final : public ForeignIOCompTaskNode {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ForeignOutputCompTaskNode);
  ForeignOutputCompTaskNode() = default;
  ~ForeignOutputCompTaskNode() override = default;

  TaskType GetTaskType() const override { return TaskType::kForeignOutput; }
};

REGISTER_INDEPENDENT_THREAD_NUM(TaskType::kForeignInput, 1);

REGISTER_SYSTEM_OP_COMP_TASK_NODE_TYPE(OperatorConf::kForeignInputConf, ForeignInputCompTaskNode);

REGISTER_INDEPENDENT_THREAD_NUM(TaskType::kForeignOutput, 1);

REGISTER_SYSTEM_OP_COMP_TASK_NODE_TYPE(OperatorConf::kForeignOutputConf, ForeignOutputCompTaskNode);

}  // namespace oneflow
