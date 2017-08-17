#include "oneflow/core/graph/loss_record_task_graph.h"
#include "oneflow/core/graph/loss_record_comp_task_node.h"

namespace oneflow {

LossRecordTaskGraph::LossRecordTaskGraph(
    const std::string& name,
    const std::vector<TaskNode*>& sorted_loss_acc_task) {
  mut_name() = name;
  BuildTaskGraph(sorted_loss_acc_task);
  BuildExecAndEnrollLbn2Regsts();
}

void LossRecordTaskGraph::BuildTaskGraph(
    const std::vector<TaskNode*>& sorted_loss_acc_task) {
  // faker_pr_conf
  ParallelConf faker_pr_conf;
  faker_pr_conf.set_policy(kFakerLossRecord);
  for (TaskNode* task : sorted_loss_acc_task) {
    auto loss_acc_task = static_cast<CompTaskNode*>(task);
    faker_pr_conf.add_device_name(loss_acc_task->device_name());
    sorted_loss_acc_tasks_.push_back(loss_acc_task);
  }
  // faker chain
  auto chain_gph = of_make_unique<ChainGraph>();
  ChainNode* faker_chain = chain_gph->NewNode();
  faker_chain->mut_op_vec() = {};
  faker_chain->mut_parallel_desc().reset(new ParallelDesc(faker_pr_conf));
  faker_chain->mut_output_lbns() = {kPackedBlobName};
  // loss_record_pr_conf
  ParallelConf loss_record_pr_conf;
  loss_record_pr_conf.set_policy(kDataParallel);
  loss_record_pr_conf.add_device_name(
      IDMgr::Singleton()->MachineName4MachineId(0) + ":persistence");
  // loss record op
  OperatorConf op_conf;
  op_conf.set_name("loss_record_" + NewUniqueId());
  op_conf.mutable_loss_record_conf();
  auto loss_record_op = OpMgr::Singleton()->AddOp(op_conf);
  // loss record chain
  ChainNode* loss_record_chain = chain_gph->NewNode();
  loss_record_chain->mut_op_vec() = {loss_record_op};
  loss_record_chain->mut_parallel_desc().reset(
      new ParallelDesc(loss_record_pr_conf));
  loss_record_chain->mut_input_lbns() = {kPackedBlobName};
  //
  Connect(faker_chain, chain_gph->NewEdge(), loss_record_chain);
  chain_gph->UpdateSourceAndSink();
  chain_gph->ToDotWithAutoFilePath();
  BuildFromChainGph<LossRecordCompTaskNode>(std::move(chain_gph), false);
}

}  // namespace oneflow
