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
#include "oneflow/core/graph/logical_node.h"

namespace oneflow {

class PackCompTaskNode final : public CompTaskNode {
 public:
  OF_DISALLOW_COPY_AND_MOVE(PackCompTaskNode);
  PackCompTaskNode() = default;
  ~PackCompTaskNode() override = default;

  TaskType GetTaskType() const override { return TaskType::kPack; }

  void ProduceAllRegstsAndBindEdges() override;
  void ConsumeAllRegsts() override;

 private:
  void BuildExecGphAndRegst() override;
  void InferProducedDataRegstTimeShape() override;
};

void PackCompTaskNode::ProduceAllRegstsAndBindEdges() {
  ProduceRegst("out", false);
  ForEachOutDataEdge([&](TaskEdge* edge) { BindEdgeWithProducedRegst(edge, "out"); });
}

void PackCompTaskNode::ConsumeAllRegsts() { ConsumeRegst("in", SoleInDataEdge()->GetSoleRegst()); }

void PackCompTaskNode::BuildExecGphAndRegst() {
  std::shared_ptr<const Operator> op = logical_node()->SoleOp();
  ExecNode* exec_node = mut_exec_gph().NewNode();
  exec_node->mut_op() = op;
  std::shared_ptr<RegstDesc> in_regst = GetSoleConsumedRegst("in");
  exec_node->BindBnWithRegst(op->SoleIbn(), in_regst);

  std::shared_ptr<RegstDesc> out_regst = GetProducedRegst("out");
  out_regst->AddLbi(op->BnInOp2Lbi(op->SoleObn()));
  exec_node->BindBnWithRegst(op->SoleObn(), out_regst);

  exec_node->InferBlobDescs(parallel_ctx());
}

void PackCompTaskNode::InferProducedDataRegstTimeShape() {
  auto TimeShape4Ibn = [&](const std::string& ibn) -> const Shape* {
    return GetSoleConsumedRegst("in")->data_regst_time_shape().get();
  };
  std::shared_ptr<Shape> time_shape(new Shape());
  logical_node()->SoleOp()->InferOutputBlobTimeShape(TimeShape4Ibn, parallel_ctx(),
                                                     time_shape.get());
  ForEachProducedDataRegst([time_shape](const std::string& name, RegstDesc* regst) {
    *regst->mut_data_regst_time_shape() = time_shape;
  });
}

REGISTER_USER_OP_COMP_TASK_NODE_TYPE("pack", PackCompTaskNode);
REGISTER_USER_OP_INDEPENDENT_AREA_ID("pack")

REGISTER_COMPUTE_TASK_NODE_STREAM_INDEX_GETTER(DeviceType::kGPU, TaskType::kPack)
    .SetStreamIndexGetterFn([](DeviceId device_id) -> uint32_t {
      auto* cuda_stream_index_generator = dynamic_cast<CudaStreamIndexGenerator*>(
          Global<IDMgr>::Get()->GetStreamIndexGeneratorManager()->GetGenerator(device_id));
      CHECK_NOTNULL(cuda_stream_index_generator);
      return cuda_stream_index_generator->GenerateComputeStreamIndex();
    });

REGISTER_COMPUTE_TASK_NODE_STREAM_INDEX_GETTER(DeviceType::kCPU, TaskType::kPack)
    .SetStreamIndexGetterFn([](DeviceId device_id) -> uint32_t {
      auto* cpu_stream_index_generator = dynamic_cast<CPUStreamIndexGenerator*>(
          Global<IDMgr>::Get()->GetStreamIndexGeneratorManager()->GetGenerator(device_id));
      CHECK_NOTNULL(cpu_stream_index_generator);
      return cpu_stream_index_generator->GenerateComputeStreamIndex();
    });

}  // namespace oneflow
