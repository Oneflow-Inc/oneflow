#include "oneflow/core/graph/local_gpu_peer_boxing_compute_task_node.h"
#include "oneflow/core/graph/logical_node.h"

namespace oneflow {

void LocalGpuPeerBoxingCompTaskNode::ProduceAllRegstsAndBindEdges() {
  this->SoleOutDataEdge()->AddRegst("out", ProduceRegst("out", false, 1, 1));
  ProduceRegst("fw_buf", false, 1, 1);
}

void LocalGpuPeerBoxingCompTaskNode::ConsumeAllRegsts() {
  std::vector<TaskEdge*> in_data_edges;
  ForEachInDataEdge([&](TaskEdge* edge) { in_data_edges.push_back(edge); });
  std::sort(in_data_edges.begin(), in_data_edges.end(),
            [&](const TaskEdge* lhs, const TaskEdge* rhs) {
              return lhs->src_node()->parallel_ctx()->parallel_id()
                     < rhs->src_node()->parallel_ctx()->parallel_id();
            });
  FOR_RANGE(int64_t, i, 0, in_data_edges.size()) {
    CHECK_EQ(i, in_data_edges.at(i)->src_node()->parallel_ctx()->parallel_id());
    ConsumeRegst("in_" + std::to_string(i), in_data_edges.at(i)->GetSoleRegst());
  }
}

void LocalGpuPeerBoxingCompTaskNode::BuildExecGphAndRegst() {
  ExecNode* node = mut_exec_gph().NewNode();
  std::shared_ptr<Operator> boxing_op = this->logical_node()->SoleOp();
  node->mut_op() = boxing_op;
  FOR_RANGE(size_t, i, 0, boxing_op->input_bns().size()) {
    const std::string& ibn = boxing_op->input_bns().Get(i);
    CHECK_EQ(GenUnRepeatedBn(ibn).second, i);
    node->BindBnWithRegst(ibn, GetSoleConsumedRegst("in_" + std::to_string(i)));
  }
  std::shared_ptr<RegstDesc> out_regst = GetProducedRegst("out");
  out_regst->AddLbi(boxing_op->BnInOp2Lbi(boxing_op->SoleObn()));
  node->BindBnWithRegst(boxing_op->SoleObn(), out_regst);
  node->AddBnToRegstAndBindIt(&Operator::fw_buf_bns, GetProducedRegst("fw_buf"));
  node->InferBlobDescs(parallel_ctx());
}

void LocalGpuPeerBoxingCompTaskNode::InferProducedDataRegstTimeShape() {
  NaiveInferProducedDataRegstTimeShape();
}

}  // namespace oneflow
