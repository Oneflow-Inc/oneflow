#include "oneflow/core/graph/nccl_boxing_task_node.h"

namespace oneflow {

void NcclBoxingTaskNode::Init(int64_t machine_id, int64_t dev_phy_id,
                              const ParallelContext& parallel_ctx, const LogicalBlobId& lbi) {
  set_machine_id(machine_id);
  set_thrd_id(Global<IDMgr>::Get()->GetGpuNcclThrdId(dev_phy_id));
  set_area_id(AreaType::kMdUpdtArea);
  parallel_ctx_ = parallel_ctx;
  lbi_ = lbi;
}

void NcclBoxingTaskNode::ProduceAllRegstsAndBindEdges() {
  this->SoleOutDataEdge()->AddRegst("out", ProduceRegst("out", false, 1, 1));
}

void NcclBoxingTaskNode::ConsumeAllRegsts() {
  ConsumeRegst("in", SoleInDataEdge()->GetSoleRegst());
}

void NcclBoxingTaskNode::BuildExecGphAndRegst() {
  ExecNode* node = mut_exec_gph().NewNode();
  std::shared_ptr<Operator> boxing_op = NewNcclBoxingOp();
  node->mut_op() = boxing_op;
  node->BindBnWithRegst(boxing_op->SoleIbn(), GetSoleConsumedRegst("in"));
  std::shared_ptr<RegstDesc> out_regst = GetProducedRegst("out");
  out_regst->AddLbi(boxing_op->BnInOp2Lbi(boxing_op->SoleObn()));
  node->BindBnWithRegst(boxing_op->SoleObn(), out_regst);
  node->InferBlobDescs(parallel_ctx());
}

void NcclBoxingTaskNode::InferProducedDataRegstTimeShape() {
  NaiveInferProducedDataRegstTimeShape();
}

void NcclBoxingTaskNode::ToProto(TaskProto* task_proto) {
  TaskNode::ToProto(task_proto);
  *(task_proto->mutable_parallel_ctx()) = parallel_ctx_;
}

std::shared_ptr<Operator> NcclBoxingReduceScatterTaskNode::NewNcclBoxingOp() const {
  OperatorConf op_conf{};
  op_conf.set_name("System-Boxing-NcclBoxingReduceScatter-" + NewUniqueId());
  op_conf.set_device_type(device_type());
  NcclBoxingReduceScatterOpConf* boxing_conf = op_conf.mutable_nccl_boxing_reduce_scatter_conf();
  *(boxing_conf->mutable_lbi()) = GetLbi();
  return ConstructOp(op_conf, &GlobalJobDesc());
}

std::shared_ptr<Operator> NcclBoxingAllGatherTaskNode::NewNcclBoxingOp() const {
  OperatorConf op_conf{};
  op_conf.set_name("System-Boxing-NcclBoxingAllGather-" + NewUniqueId());
  op_conf.set_device_type(device_type());
  NcclBoxingAllGatherOpConf* boxing_conf = op_conf.mutable_nccl_boxing_all_gather_conf();
  *(boxing_conf->mutable_lbi()) = GetLbi();
  return ConstructOp(op_conf, &GlobalJobDesc());
}

std::shared_ptr<Operator> NcclBoxingAllReduceTaskNode::NewNcclBoxingOp() const {
  OperatorConf op_conf{};
  op_conf.set_name("System-Boxing-NcclBoxingAllReduce-" + NewUniqueId());
  op_conf.set_device_type(device_type());
  NcclBoxingAllReduceOpConf* boxing_conf = op_conf.mutable_nccl_boxing_all_reduce_conf();
  *(boxing_conf->mutable_lbi()) = GetLbi();
  return ConstructOp(op_conf, &GlobalJobDesc());
}

}  // namespace oneflow
