#include "oneflow/core/graph/record_load_compute_task_node.h"
#include "oneflow/core/graph/decode_compute_task_node.h"
#include "oneflow/core/graph/logical_node.h"
#include "oneflow/core/common/str_util.h"
#include "oneflow/core/common/balanced_splitter.h"

namespace oneflow {

void RecordLoadCompTaskNode::ProduceAllRegstsAndBindEdges() {
  std::shared_ptr<RegstDesc> record_regst = ProduceRegst("record", false, 2, 2);
  for (TaskEdge* edge : out_edges()) { edge->AddRegst("record", record_regst); }
}

void RecordLoadCompTaskNode::BuildExecGphAndRegst() {
  std::shared_ptr<RegstDesc> record_regst = GetProducedRegst("record");
  ExecNode* node = mut_exec_gph().NewNode();
  node->mut_op() = logical_node()->SoleOp();
  for (const std::string& obn : node->op()->output_bns()) {
    const LogicalBlobId& lbi = node->op()->BnInOp2Lbi(obn);
    record_regst->AddLbi(lbi);
    node->BindBnWithRegst(obn, record_regst);
  }
  node->InferBlobDescs(parallel_ctx());
}

void RecordLoadCompTaskNode::InferProducedRegstTimeShape() {
  std::shared_ptr<Shape> time_shape = std::make_shared<Shape>(std::vector<int64_t>(
      {Global<JobDesc>::Get()->TotalBatchNum(), Global<JobDesc>::Get()->NumOfPiecesInBatch()}));
  for (auto& pair : produced_regsts()) { pair.second->mut_time_shape() = time_shape; }
}

}  // namespace oneflow
