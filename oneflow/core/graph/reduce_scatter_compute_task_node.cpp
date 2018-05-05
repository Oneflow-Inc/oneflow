#include "oneflow/core/graph/reduce_scatter_compute_task_node.h"
#include "oneflow/core/graph/logical_node.h"

namespace oneflow {

void ReduceScatterCompTaskNode::ProduceAllRegstsAndBindEdges() {
  for (TaskEdge* edge : out_edges()) {
    TaskNode* dst_node = edge->dst_node();
    while (dst_node->GetTaskType() != TaskType::kReduceAdd) {
      dst_node = dst_node->SoleOutEdge()->dst_node();
    }
    CompTaskNode* reduce_add_node = static_cast<CompTaskNode*>(dst_node);
    std::string out_regst_name = "out_" + std::to_string(reduce_add_node->parallel_id());
    std::shared_ptr<RegstDesc> out_regst = ProduceRegst(out_regst_name);
    edge->AddRegst(out_regst_name, out_regst);
    if (this->parallel_id() == reduce_add_node->parallel_id()
        && device_type() == DeviceType::kGPU) {
      MemoryCase* mem_case = out_regst.get()->mut_mem_case();
      mem_case->Clear();
      mem_case->mutable_device_cuda_mem()->set_device_id(
          Global<IDMgr>::Get()->GetGpuDevPhyIdFromThrdId(thrd_id()));
    }
  }
}

void ReduceScatterCompTaskNode::ConsumeAllRegsts() {
  ConsumeRegst("in", this->SoleInEdge()->GetSoleRegst());
}

void ReduceScatterCompTaskNode::InitProducedRegstMemCase(MemoryCase* mem_case) {
  mem_case->mutable_host_mem();
  if (device_type() == DeviceType::kGPU) { mem_case->mutable_host_mem()->set_used_by_device(true); }
}

void ReduceScatterCompTaskNode::BuildExecGphAndRegst() {
  ExecNode* node = mut_exec_gph().NewNode();
  std::shared_ptr<Operator> reduce_scatter_op = this->logical_node()->SoleOp();
  node->mut_op() = reduce_scatter_op;
  node->BindBnWithRegst(reduce_scatter_op->SoleIbn(), GetSoleConsumedRegst("in"));
  FOR_RANGE(size_t, i, 0, reduce_scatter_op->output_bns().size()) {
    std::shared_ptr<RegstDesc> out_regst = GetProducedRegst("out_" + std::to_string(i));
    const std::string& obn = reduce_scatter_op->output_bns().Get(i);
    out_regst->AddLbi(reduce_scatter_op->BnInOp2Lbi(obn));
    node->BindBnWithRegst(obn, out_regst);
  }
  node->InferBlobDescs(parallel_ctx());
}

}  // namespace oneflow
