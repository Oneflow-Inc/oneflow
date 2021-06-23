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
#include "oneflow/core/common/protobuf.h"

namespace oneflow {

class EsacCompTaskNode final : public CompTaskNode {
 public:
  OF_DISALLOW_COPY_AND_MOVE(EsacCompTaskNode);
  EsacCompTaskNode() = default;
  ~EsacCompTaskNode() override = default;

  void ProduceAllRegstsAndBindEdges() override;
  void ConsumeAllRegsts() override;

  TaskType GetTaskType() const override { return TaskType::kEsac; }
  CudaWorkType GetCudaWorkType() const override {
#ifdef WITH_CUDA
    return CudaWorkType::kCompute;
#else
    UNIMPLEMENTED();
#endif
  }

 private:
  void BuildExecGphAndRegst() override;
  void InferProducedDataRegstTimeShape() override;
  bool IsIndependent() const override { return true; }
};

void EsacCompTaskNode::ConsumeAllRegsts() {
  HashMap<LogicalBlobId, int64_t> lbi2ibn_id;
  FOR_RANGE(int64_t, ibn_id, 0, op()->input_bns().size()) {
    CHECK(lbi2ibn_id.emplace(op()->BnInOp2Lbi(GenRepeatedBn("in", ibn_id)), ibn_id).second);
  }
  ForEachInDataEdge([&](TaskEdge* edge) {
    const OpNode* pred = GetOnePredOpNodeOnEdge(edge);
    int64_t ibn_id = -1;
    for (const std::string& obn : pred->shared_op()->output_bns()) {
      const LogicalBlobId& lbi = pred->shared_op()->BnInOp2Lbi(obn);
      if (lbi2ibn_id.find(lbi) != lbi2ibn_id.cend()) {
        CHECK_EQ(ibn_id, -1);
        ibn_id = lbi2ibn_id.at(lbi);
      }
    }
    CHECK_NE(ibn_id, -1);
    ConsumeRegst("in_" + std::to_string(ibn_id), edge->GetSoleRegst());
  });
}

void EsacCompTaskNode::ProduceAllRegstsAndBindEdges() {
  std::shared_ptr<RegstDesc> out_regst = ProduceRegst("out", false, 1, 1);
  ForEachOutDataEdge([&](TaskEdge* edge) { edge->AddRegst("out", out_regst); });
}

void EsacCompTaskNode::BuildExecGphAndRegst() {
  ExecNode* node = mut_exec_gph().NewNode();
  std::shared_ptr<const Operator> sole_op = this->op();
  node->mut_op() = sole_op;
  FOR_RANGE(int64_t, ibn_id, 0, sole_op->input_bns().size()) {
    node->BindBnWithRegst(GenRepeatedBn("in", ibn_id),
                          GetSoleConsumedRegst("in_" + std::to_string(ibn_id)));
  }
  std::shared_ptr<RegstDesc> out_regst = GetProducedRegst("out");
  out_regst->AddLbi(sole_op->BnInOp2Lbi("out"));
  node->BindBnWithRegst("out", out_regst);
  node->InferBlobDescs(parallel_ctx());
}

void EsacCompTaskNode::InferProducedDataRegstTimeShape() { NaiveInferProducedDataRegstTimeShape(); }

REGISTER_TICK_TOCK_TASK_TYPE(TaskType::kEsac);

REGISTER_COMPUTE_TASK_NODE_STREAM_INDEX_GETTER(DeviceType::kCPU, TaskType::kEsac)
    .SetStreamIndexGetterFn([](CPUStreamIndexGenerator* generator) -> uint32_t {
      return generator->GenerateTickTockStreamIndex();
    });

REGISTER_SYSTEM_OP_COMP_TASK_NODE_TYPE(OperatorConf::kEsacConf, EsacCompTaskNode);

}  // namespace oneflow
