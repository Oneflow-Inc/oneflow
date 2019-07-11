#include "oneflow/core/graph/cuda_copy_peer_task_node.h"
#include "oneflow/core/operator/operator.h"
#include "oneflow/core/job/thrd_id_generator.h"

namespace oneflow {

void CudaCopyPeerTaskNode::ProduceAllRegstsAndBindEdges() {
  std::shared_ptr<RegstDesc> out_regst = ProduceRegst("copy_out", false, 1, 1);
  ForEachOutDataEdge([&](TaskEdge* edge) { edge->AddRegst("copy_out", out_regst); });
}

void CudaCopyPeerTaskNode::ConsumeAllRegsts() {
  ConsumeRegst("copy_in", SoleInDataEdge()->GetSoleRegst());
}

void CudaCopyPeerTaskNode::BuildExecGphAndRegst() {
  auto out_regst = GetProducedRegst("copy_out");
  auto in_regst = GetSoleConsumedRegst("copy_in");
  out_regst->CopyBlobDescFrom(in_regst.get());
  ExecNode* node = mut_exec_gph().NewNode();
  node->mut_op() = ConstructOp(NewCopyOpConf());
  node->BindBnWithRegst(node->op()->SoleIbn(), in_regst);
  node->BindBnWithRegst(node->op()->SoleObn(), out_regst);
}

void CudaCopyPeerTaskNode::InferProducedDataRegstTimeShape() {
  NaiveInferProducedDataRegstTimeShape();
}

void CudaCopyPeerTaskNode::Init(int64_t machine_id, int64_t dev_phy_id) {
  set_machine_id(machine_id);
  if (dev_phy_id % 2 == 0) {
    set_thrd_id(Global<IDMgr>::Get()->GetGpuD2DThrdId(dev_phy_id));
  } else {
    set_thrd_id(Global<IDMgr>::Get()->GetGpuNcclScatterThrdId(dev_phy_id));
  }
}

OperatorConf CudaCopyPeerTaskNode::NewCopyOpConf() {
  OperatorConf conf{};
  conf.set_name("copy_p2p_" + NewUniqueId());
  conf.set_device_type(device_type());
  conf.mutable_cuda_copy_peer_conf();
  return conf;
}

}  // namespace oneflow
