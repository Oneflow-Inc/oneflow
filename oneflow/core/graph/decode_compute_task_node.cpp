#include "oneflow/core/graph/decode_compute_task_node.h"
#include "oneflow/core/graph/task_graph.h"
#include "oneflow/core/graph/logical_node.h"

namespace oneflow {

void DecodeCompTaskNode::ProduceAllRegstsAndBindEdges() {
  ProduceRegst("data_tmp", 1, 1);
  ProduceRegst("boxing_out");
  ProduceRegst("121_out");
  for (TaskEdge* edge : out_edges()) {
    const LogicalNode* succ_logical = GetOneSuccLogicalNodeOnEdge(edge);
    BldSubTskGphMthd mthd = GetMthdForBldSubTskGph(logical_node(), succ_logical);
    if (mthd == &TaskGraph::BldSubTskGphByBoxing) {
      BindEdgeWithProducedRegst(edge, "boxing_out");
    } else if (mthd == &TaskGraph::BldSubTskGphByOneToOne) {
      BindEdgeWithProducedRegst(edge, "121_out");
    } else {
      UNIMPLEMENTED();
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
  std::shared_ptr<RegstDesc> out_regst_boxing = GetProducedRegst("boxing_out");
  std::shared_ptr<RegstDesc> out_regst_121 = GetProducedRegst("121_out");
  std::shared_ptr<RegstDesc> data_tmp_regst = GetProducedRegst("data_tmp");
  ExecNode* node = mut_exec_gph().NewNode();
  node->mut_op() = logical_node()->SoleOp();

  const HashSet<LogicalBlobId>& lbi_boxing = logical_node()->lbi_boxing();
  const HashSet<LogicalBlobId>& lbi_121 = logical_node()->lbi_121();
  for (const std::string& obn : node->op()->output_bns()) {
    const LogicalBlobId& lbi = node->op()->BnInOp2Lbi(obn);
    if (lbi_boxing.find(lbi) != lbi_boxing.end()) {
      out_regst_boxing->AddLbi(lbi);
      node->BindBnWithRegst(obn, out_regst_boxing);
    } else if (lbi_121.find(lbi) != lbi_121.end()) {
      out_regst_121->AddLbi(lbi);
      node->BindBnWithRegst(obn, out_regst_121);
    } else {
      data_tmp_regst->AddLbi(lbi);
      node->BindBnWithRegst(obn, data_tmp_regst);
    }
  }
  node->AddBnToRegstAndBindIt(&Operator::data_tmp_bns, data_tmp_regst);
  node->InferBlobDescs(parallel_ctx());
}

}  // namespace oneflow
