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

class CallbackNotifyCompTaskNode final : public CompTaskNode {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CallbackNotifyCompTaskNode);
  CallbackNotifyCompTaskNode() = default;
  ~CallbackNotifyCompTaskNode() = default;

  TaskType GetTaskType() const override { return TaskType::kCallbackNotify; }
  bool IsIndependent() const override { return true; }

 private:
  void ProduceAllRegstsAndBindEdges() override;
  void ConsumeAllRegsts() override;
  void BuildExecGphAndRegst() override;
};

void CallbackNotifyCompTaskNode::ProduceAllRegstsAndBindEdges() {}

void CallbackNotifyCompTaskNode::ConsumeAllRegsts() {
  ForEachInDataEdge([&](TaskEdge* edge) { ConsumeRegst("in", edge->GetSoleRegst()); });
}

void CallbackNotifyCompTaskNode::BuildExecGphAndRegst() {
  ExecNode* node = mut_exec_gph().NewNode();
  node->mut_op() = this->op();
  for (const std::string& ibn : node->op()->input_bns()) {
    node->BindBnWithOneOfTheRegsts(ibn, GetConsumedRegst("in"));
  }
  CHECK(node->op()->tmp_bns().empty());
  CHECK(node->op()->output_bns().empty());
}

REGISTER_INDEPENDENT_THREAD_NUM(TaskType::kCallbackNotify, 1);

REGISTER_COMPUTE_TASK_NODE_STREAM_INDEX_GETTER(DeviceType::kCPU, TaskType::kCallbackNotify)
    .SetStreamIndexGetterFn([](CPUStreamIndexGenerator* generator) -> uint32_t {
      return generator->GenerateIndependentTaskStreamIndex(TaskType::kCallbackNotify);
    });

REGISTER_SYSTEM_OP_COMP_TASK_NODE_TYPE(OperatorConf::kCallbackNotifyConf,
                                       CallbackNotifyCompTaskNode);

}  // namespace oneflow
