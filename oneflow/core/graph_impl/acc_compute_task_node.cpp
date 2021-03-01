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
#include "oneflow/core/graph/accumulate_compute_task_node.h"
#include "oneflow/core/graph/logical_node.h"
#include "oneflow/core/framework/framework.h"

namespace oneflow {

class AccCompTaskNode final : public AccumulateCompTaskNode {
 public:
  OF_DISALLOW_COPY_AND_MOVE(AccCompTaskNode);
  AccCompTaskNode() = default;
  ~AccCompTaskNode() = default;
  TaskType GetTaskType() const override { return TaskType::kAcc; }
  void BuildExecGphAndRegst() override;

 private:
  void InferProducedDataRegstTimeShape() override;
};

void AccCompTaskNode::InferProducedDataRegstTimeShape() {
  auto TimeShape4Ibn = [&](const std::string& ibn) -> const Shape* {
    return GetSoleConsumedRegst("one")->data_regst_time_shape().get();
  };
  std::shared_ptr<Shape> time_shape(new Shape());
  logical_node()->SoleOp()->InferOutputBlobTimeShape(TimeShape4Ibn, parallel_ctx(),
                                                     time_shape.get());
  ForEachProducedDataRegst([time_shape](const std::string& name, RegstDesc* regst) {
    *regst->mut_data_regst_time_shape() = time_shape;
  });
}

void AccCompTaskNode::BuildExecGphAndRegst() {
  std::shared_ptr<RegstDesc> one_regst = GetSoleConsumedRegst("one");
  std::shared_ptr<RegstDesc> acc_regst = GetProducedRegst("acc");
  std::shared_ptr<const Operator> op = logical_node()->SoleOp();
  ExecNode* exec_node = mut_exec_gph().NewNode();
  exec_node->mut_op() = op;
  exec_node->BindBnWithRegst(op->SoleIbn(), one_regst);
  acc_regst->AddLbi(op->BnInOp2Lbi(op->SoleObn()));
  exec_node->BindBnWithRegst(op->SoleObn(), acc_regst);
  exec_node->InferBlobDescs(parallel_ctx());
  acc_regst->ForEachLbi([acc_regst](const LogicalBlobId& lbi) {
    const BlobDesc* blob_desc = acc_regst->GetBlobDesc(lbi);
    CHECK_EQ(blob_desc->is_dynamic(), false);
    CHECK_EQ(blob_desc->is_tensor_list(), false);
  });
}

REGISTER_USER_OP_COMP_TASK_NODE_TYPE("acc", AccCompTaskNode);
REGISTER_USER_OP_INDEPENDENT_AREA_ID("acc")

REGISTER_COMPUTE_TASK_NODE_STREAM_INDEX_GETTER(DeviceType::kGPU, TaskType::kAcc)
    .SetStreamIndexGetterFn([](DeviceId device_id) -> uint32_t {
      auto* cuda_stream_index_generator = dynamic_cast<CudaStreamIndexGenerator*>(
          Global<IDMgr>::Get()->GetStreamIndexGeneratorManager()->GetGenerator(device_id));
      CHECK_NOTNULL(cuda_stream_index_generator);
      return cuda_stream_index_generator->GenerateComputeStreamIndex();
    });

REGISTER_COMPUTE_TASK_NODE_STREAM_INDEX_GETTER(DeviceType::kCPU, TaskType::kAcc)
    .SetStreamIndexGetterFn([](DeviceId device_id) -> uint32_t {
      auto* cpu_stream_index_generator = dynamic_cast<CPUStreamIndexGenerator*>(
          Global<IDMgr>::Get()->GetStreamIndexGeneratorManager()->GetGenerator(device_id));
      CHECK_NOTNULL(cpu_stream_index_generator);
      return cpu_stream_index_generator->GenerateComputeStreamIndex();
    });

}  // namespace oneflow
