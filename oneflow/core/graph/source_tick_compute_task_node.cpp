#include "oneflow/core/graph/source_tick_compute_task_node.h"
#include "oneflow/core/graph/decode_compute_task_node.h"
#include "oneflow/core/graph/logical_node.h"
#include "oneflow/core/common/str_util.h"
#include "oneflow/core/common/balanced_splitter.h"

namespace oneflow {

void SourceTickCompTaskNode::ProduceAllRegstsAndBindEdges() {
  std::shared_ptr<RegstDesc> out_regst = ProduceRegst("out", false, 2, 2);
  ForEachOutDataEdge([&](TaskEdge* edge) { edge->AddRegst("out", out_regst); });
}

void SourceTickCompTaskNode::BuildExecGphAndRegst() {
  std::shared_ptr<RegstDesc> out_regst = GetProducedRegst("out");
  ExecNode* node = mut_exec_gph().NewNode();
  node->mut_op() = logical_node()->SoleOp();
  for (const std::string& obn : node->op()->output_bns()) {
    const LogicalBlobId& lbi = node->op()->BnInOp2Lbi(obn);
    out_regst->AddLbi(lbi);
    node->BindBnWithRegst(obn, out_regst);
  }
  node->InferBlobDescs(parallel_ctx());
}

void SourceTickCompTaskNode::InferProducedDataRegstTimeShape() {
  std::shared_ptr<Shape> time_shape(
      new Shape({GlobalJobDesc().TotalBatchNum(), GlobalJobDesc().NumOfPiecesInBatch()}));
  ForEachProducedDataRegst([time_shape](const std::string& name, RegstDesc* regst) {
    *regst->mut_data_regst_time_shape() = time_shape;
  });
}

}  // namespace oneflow
