#include "graph/model_save_task_graph.h"

namespace oneflow {

class MdSaveCompTaskNode;

MdSaveTaskGraph::MdSaveTaskGraph(const std::string& name,
                                 CompTaskNode* update_task,
                                 const std::string& dot_path_prefix) {
  mut_name() = name;
  update_task_ = update_task;
  BuildTaskGraph(dot_path_prefix);
  BuildExecAndEnrollLbn2Regsts();
}

void MdSaveTaskGraph::BuildTaskGraph(const std::string& dot_path_prefix) {
  auto chain_gph = of_make_unique<ChainGraph> ();
  // faker
  ChainNode* faker_chain = chain_gph->NewNode();
  ParallelConf faker_pr_conf;
  faker_pr_conf.set_policy(kDataParallel);
  faker_pr_conf.mutable_device_set()->add_device_name(update_task_->device_name());
  faker_chain->mut_parallel_desc().reset(new ParallelDesc(faker_pr_conf));
  faker_chain->mut_output_lbns() = {RegstDesc::kAllLbn};
  // save
  ChainNode* save_chain = chain_gph->NewNode();
  std::string machine_name = 
      GetMachineNameFromDeviceName(update_task_->device_name());
  ParallelConf save_pr_conf;
  save_pr_conf.set_policy(kDataParallel);
  save_pr_conf.mutable_device_set()->add_device_name(machine_name + ":disk");
  save_chain->mut_parallel_desc().reset(new ParallelDesc(save_pr_conf));
  save_chain->mut_input_lbns() = {RegstDesc::kAllLbn};
  //
  Connect(faker_chain, chain_gph->NewEdge(), save_chain);
  chain_gph->UpdateSourceAndSink();
  chain_gph->ToDotFile(dot_path_prefix + "chain_graph.dot");
  BuildFromChainGph<MdSaveCompTaskNode>(std::move(chain_gph), false, dot_path_prefix);
}

} // namespace oneflow
