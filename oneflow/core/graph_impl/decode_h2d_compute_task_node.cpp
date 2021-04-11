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
#include "oneflow/core/graph/normal_forward_compute_task_node.h"
#include "oneflow/core/graph/compute_task_node.h"
#include "oneflow/core/framework/to_string.h"

namespace oneflow {

class DecodeH2DCompTaskNode final : public CompTaskNode {
 public:
  OF_DISALLOW_COPY_AND_MOVE(DecodeH2DCompTaskNode);
  DecodeH2DCompTaskNode() = default;
  ~DecodeH2DCompTaskNode() override = default;

  void ProduceAllRegstsAndBindEdges() override;
  void ConsumeAllRegsts() override;

  TaskType GetTaskType() const override { return TaskType::kDecodeH2D; }
  CudaWorkType GetCudaWorkType() const override {
#ifdef WITH_CUDA
    return CudaWorkType::kDecodeH2D;
#else
    UNIMPLEMENTED();
#endif
  }

 private:
  void BuildExecGphAndRegst() override;
};

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
  std::shared_ptr<const Operator> sole_op = op();
  node->mut_op() = sole_op;
  node->BindBnWithRegst(sole_op->SoleIbn(), GetSoleConsumedRegst("in"));
  std::shared_ptr<RegstDesc> out_regst = GetProducedRegst("out");
  out_regst->AddLbi(sole_op->BnInOp2Lbi(sole_op->SoleObn()));
  node->BindBnWithRegst(sole_op->SoleObn(), out_regst);
  node->AddBnToRegstAndBindIt(&Operator::tmp_bns, GetProducedRegst("tmp"));
  node->InferBlobDescs(parallel_ctx());
}

#ifdef WITH_CUDA

REGISTER_COMPUTE_TASK_NODE_STREAM_INDEX_GETTER(DeviceType::kGPU, TaskType::kDecodeH2D)
    .SetStreamIndexGetterFn([](CudaStreamIndexGenerator* generator) -> uint32_t {
      return generator->GenerateDecodeH2DStreamIndex();
    });

#endif

namespace {

CompTaskNode* CreateCompTaskNodeByOpDeviceType(const OperatorConf& op_conf) {
  if (CHECK_JUST(DeviceType4DeviceTag(op_conf.device_tag())) == DeviceType::kGPU) {
    return new DecodeH2DCompTaskNode;
  } else {
    return new NormalForwardCompTaskNode;
  }
}

}  // namespace

REGISTER_SYSTEM_OP_COMP_TASK_NODE_TYPE_WITH_FUNC(OperatorConf::kImageDecoderRandomCropResizeConf,
                                                 CreateCompTaskNodeByOpDeviceType);

}  // namespace oneflow
