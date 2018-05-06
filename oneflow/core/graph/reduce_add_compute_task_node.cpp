#include "oneflow/core/graph/reduce_add_compute_task_node.h"
#include "oneflow/core/graph/logical_node.h"

namespace oneflow {

void ReduceAddCompTaskNode::ProduceAllRegstsAndBindEdges() {
  for (TaskEdge* edge : out_edges()) {
    TaskNode* dst_node = edge->dst_node();
    while (dst_node->GetTaskType() != TaskType::kReduceGather) {
      dst_node = dst_node->SoleOutEdge()->dst_node();
    }
    CompTaskNode* reduce_gather_node = static_cast<CompTaskNode*>(dst_node);
    std::string out_regst_name = "out_" + std::to_string(reduce_gather_node->parallel_id());
    std::shared_ptr<RegstDesc> out_regst = ProduceRegst(out_regst_name);
    edge->AddRegst(out_regst_name, out_regst);
    if (this->parallel_id() == reduce_gather_node->parallel_id()
        && device_type() == DeviceType::kGPU) {
      MemoryCase* mem_case = out_regst.get()->mut_mem_case();
      mem_case->Clear();
      mem_case->mutable_device_cuda_mem()->set_device_id(
          Global<IDMgr>::Get()->GetGpuDevPhyIdFromThrdId(thrd_id()));
    }
  }
}

void ReduceAddCompTaskNode::ConsumeAllRegsts() {
  for (TaskEdge* edge : in_edges()) {
    TaskNode* src_node = edge->src_node();
    while (src_node->GetTaskType() != TaskType::kReduceScatter) {
      src_node = src_node->SoleInEdge()->src_node();
    }
    CompTaskNode* reduce_scatter_node = static_cast<CompTaskNode*>(src_node);
    std::string in_regst_name = "in_" + std::to_string(reduce_scatter_node->parallel_id());
    ConsumeRegst(in_regst_name, edge->GetSoleRegst());
  }
}

void ReduceAddCompTaskNode::InitProducedRegstMemCase(MemoryCase* mem_case) {
  mem_case->mutable_host_mem();
  if (device_type() == DeviceType::kGPU) { mem_case->mutable_host_mem()->set_used_by_device(true); }
}

void ReduceAddCompTaskNode::BuildExecGphAndRegst() {
  ExecNode* node = mut_exec_gph().NewNode();
  std::shared_ptr<Operator> reduce_add_op = this->logical_node()->SoleOp();
  node->mut_op() = reduce_add_op;
  FOR_RANGE(size_t, i, 0, reduce_add_op->input_bns().size()) {
    std::shared_ptr<RegstDesc> in_regst = GetSoleConsumedRegst("in_" + std::to_string(i));
    node->BindBnWithRegst(reduce_add_op->input_bns().Get(i), in_regst);
  }
  FOR_RANGE(size_t, i, 0, reduce_add_op->output_bns().size()) {
    std::shared_ptr<RegstDesc> out_regst = GetProducedRegst("out_" + std::to_string(i));
    const std::string& obn = reduce_add_op->output_bns().Get(i);
    out_regst->AddLbi(reduce_add_op->BnInOp2Lbi(obn));
    node->BindBnWithRegst(obn, out_regst);
  }
  node->InferBlobDescs(parallel_ctx());
}

}  // namespace oneflow
