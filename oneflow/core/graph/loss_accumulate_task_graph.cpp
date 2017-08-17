#include "oneflow/core/graph/loss_accumulate_task_graph.h"
#include "oneflow/core/graph/loss_accumulate_comp_task_node.h"

namespace oneflow {

LossAccTaskGraph::LossAccTaskGraph(const std::string& name,
                                   CompTaskNode* loss_task) {
  mut_name() = name;
  loss_task_ = loss_task;
  BuildTaskGraph();
  BuildExecAndEnrollLbn2Regsts();
}

void LossAccTaskGraph::BuildTaskGraph() {
  // loss acc op
  OperatorConf op_conf;
  op_conf.set_name("loss_acc_" + NewUniqueId());
  op_conf.mutable_accumulate_conf();
  auto loss_acc_op = OpMgr::Singleton()->AddOp(op_conf);
  // parallel_desc
  ParallelConf pr_conf;
  pr_conf.set_policy(kDataParallel);
  pr_conf.mutable_device_set()->add_device_name(loss_task_->device_name());
  auto pr_desc = std::make_shared<ParallelDesc>(pr_conf);
  // faker chain
  auto chain_gph = of_make_unique<ChainGraph>();
  ChainNode* faker_chain = chain_gph->NewNode();
  faker_chain->mut_op_vec() = {};
  faker_chain->mut_parallel_desc() = pr_desc;
  faker_chain->mut_output_lbns() = {kPackedBlobName};
  // loss acc chain
  ChainNode* loss_acc_chain = chain_gph->NewNode();
  loss_acc_chain->mut_op_vec() = {loss_acc_op};
  loss_acc_chain->mut_parallel_desc() = pr_desc;
  loss_acc_chain->mut_input_lbns() = {kPackedBlobName};
  //
  Connect(faker_chain, chain_gph->NewEdge(), loss_acc_chain);
  chain_gph->UpdateSourceAndSink();
  chain_gph->ToDotWithAutoFilePath();
  BuildFromChainGph<LossAccCompTaskNode>(std::move(chain_gph), false);
}

}  // namespace oneflow
