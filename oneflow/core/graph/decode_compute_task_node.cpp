#include "oneflow/core/graph/decode_compute_task_node.h"
#include "oneflow/core/graph/task_graph.h"
#include "oneflow/core/graph/logical_node.h"

namespace oneflow {

void DecodeCompTaskNode::ProduceAllRegstsAndBindEdges() {
  ProduceRegst("data_tmp", true, 1, 1);
  ProduceB121Regst("out");
  ProduceB121Regst("fw_pb_out");
  for (TaskEdge* edge : out_edges()) {
    BindEdgeWithProducedB121Regst(edge, "out");
    if (edge->dst_node()->GetTaskType() != TaskType::kCopyHd) {
      BindEdgeWithProducedB121Regst(edge, "fw_pb_out");
    }
  }
}

void DecodeCompTaskNode::ConsumeAllRegsts() {
  if (in_edges().size() == 1) {
    ConsumeRegst("record", SoleInEdge()->GetSoleRegst());
  } else {
    CHECK_EQ(in_edges().size(), 0);
  }
}

void DecodeCompTaskNode::BuildExecGphAndRegst() {
  std::shared_ptr<RegstDesc> data_tmp_regst = GetProducedRegst("data_tmp");
  std::weak_ptr<RegstDesc> record_regst = GetSoleConsumedRegst("record");
  ExecNode* node = mut_exec_gph().NewNode();
  node->mut_op() = logical_node()->SoleOp();
  node->BindBnWithRegst(node->op()->SoleIbn(), record_regst);
  for (const std::string& obn : node->op()->output_bns()) {
    const LogicalBlobId& lbi = node->op()->BnInOp2Lbi(obn);
    if (TryAddLbiToB121RegstAndBindIt(node, obn, (lbi.is_fw_pb() ? "fw_pb_out" : "out")) == false) {
      data_tmp_regst->AddLbi(lbi);
      node->BindBnWithRegst(obn, data_tmp_regst);
    }
  }
  node->AddBnToRegstAndBindIt(&Operator::data_tmp_bns, data_tmp_regst);
  node->InferBlobDescs(parallel_ctx());
}

}  // namespace oneflow
