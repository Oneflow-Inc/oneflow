#include "oneflow/core/graph/model_update_task_graph.h"
#include "oneflow/core/graph/model_update_comp_task_node.h"

namespace oneflow {

MdUpdtTaskGraph::MdUpdtTaskGraph(const std::string& name,
                                 CompTaskNode* diff_acc_task,
                                 const std::string& dot_path_prefix) {
  mut_name() = name;
  diff_acc_task_ = diff_acc_task;
  BuildTaskGraph(dot_path_prefix);
  BuildExecAndEnrollLbn2Regsts();
}

void MdUpdtTaskGraph::BuildTaskGraph(const std::string& dot_path_prefix) {
  auto chain_gph = of_make_unique<ChainGraph> ();
  // faker
  ChainNode* faker_chain = chain_gph->NewNode();
  ParallelConf faker_pr_conf;
  faker_pr_conf.set_policy(kDataParallel);
  faker_pr_conf.mutable_device_set()->add_device_name(diff_acc_task_->device_name());
  faker_chain->mut_parallel_desc().reset(new ParallelDesc(faker_pr_conf));
  faker_chain->mut_output_lbns() = {kBaledBlobName};
  // update
  ChainNode* updt_chain = chain_gph->NewNode();
  ParallelConf updt_pr_conf;
  updt_pr_conf.set_policy(kDataParallel);
  updt_pr_conf.mutable_device_set()->add_device_name(diff_acc_task_->device_name());
  updt_chain->mut_parallel_desc().reset(new ParallelDesc(updt_pr_conf));
  updt_chain->mut_input_lbns() = {kBaledBlobName};
  //
  Connect(faker_chain, chain_gph->NewEdge(), updt_chain);
  chain_gph->UpdateSourceAndSink();
  chain_gph->ToDotFile(dot_path_prefix + "chain_graph.dot");
  BuildFromChainGph<MdUpdtCompTaskNode>(std::move(chain_gph), false, dot_path_prefix);
  for (const auto& node : nodes()) {
    auto model_updt_comp_task_node = 
        dynamic_cast<MdUpdtCompTaskNode*>(node.get());
    if (model_updt_comp_task_node != nullptr) {
      model_updt_comp_task_node->set_related_diff_acc_task_parallel_id(diff_acc_task_->parallel_id());
    }
  }
}

} // namespace oneflow
