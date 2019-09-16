#include "oneflow/core/graph/record_load_compute_task_node.h"
#include "oneflow/core/graph/decode_compute_task_node.h"
#include "oneflow/core/graph/logical_node.h"
#include "oneflow/core/common/str_util.h"
#include "oneflow/core/common/balanced_splitter.h"

namespace oneflow {

void RecordLoadCompTaskNode::ProduceAllRegstsAndBindEdges() {
  std::shared_ptr<RegstDesc> record_regst = ProduceRegst("record", false, 2, 2);
  ForEachOutDataEdge([&](TaskEdge* edge) { edge->AddRegst("record", record_regst); });
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

void RecordLoadCompTaskNode::InferProducedDataRegstTimeShape() {
  std::shared_ptr<Shape> time_shape(new Shape(
      {Global<JobDesc>::Get()->TotalBatchNum(), Global<JobDesc>::Get()->NumOfPiecesInBatch()}));
  ForEachProducedDataRegst([time_shape](const std::string& name, RegstDesc* regst) {
    *regst->mut_data_regst_time_shape() = time_shape;
  });
}

void RecordLoadCompTaskNode::GenerateNonCtrlRegstHandlerProto(TaskProto* task_proto) const {
  RegstHandlerProto naive_proto = CreateRegstHandlerProto("Naive");
  ForEachNonCtrlConsumedRegstDescId([&](int64_t regst_desc_id) {
    naive_proto.mutable_consumed_regst_desc_ids()->add_regst_desc_id(regst_desc_id);
  });
  ForEachNonCtrlProducedRegstDescId([&](int64_t regst_desc_id) {
    naive_proto.mutable_produced_regst_desc_ids()->add_regst_desc_id(regst_desc_id);
  });
  if (!IsRegstHandlerProtoEmpty(naive_proto)) {
    *(task_proto->mutable_regst_handlers()->Add()) = naive_proto;
  }
}

}  // namespace oneflow
