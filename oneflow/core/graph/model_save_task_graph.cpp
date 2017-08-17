#include "oneflow/core/graph/model_save_task_graph.h"
#include "oneflow/core/graph/model_save_comp_task_node.h"
#include "oneflow/core/graph/model_update_comp_task_node.h"

namespace oneflow {

MdSaveTaskGraph::MdSaveTaskGraph(const std::string& name,
                                 CompTaskNode* update_task) {
  mut_name() = name;
  update_task_ = update_task;
  BuildTaskGraph();
  BuildExecAndEnrollLbn2Regsts();
}

void MdSaveTaskGraph::BuildTaskGraph() {
  auto chain_gph = of_make_unique<ChainGraph>();
  // faker
  ChainNode* faker_chain = chain_gph->NewNode();
  ParallelConf faker_pr_conf;
  faker_pr_conf.set_policy(kDataParallel);
  faker_pr_conf.add_device_name(update_task_->device_name());
  faker_chain->mut_parallel_desc().reset(new ParallelDesc(faker_pr_conf));
  faker_chain->mut_output_lbns() = {kPackedBlobName};
  // save
  ChainNode* save_chain = chain_gph->NewNode();
  std::string machine_name =
      GetMachineNameFromDeviceName(update_task_->device_name());
  ParallelConf save_pr_conf;
  save_pr_conf.set_policy(kDataParallel);
  save_pr_conf.add_device_name(machine_name + ":persistence");
  save_chain->mut_parallel_desc().reset(new ParallelDesc(save_pr_conf));
  save_chain->mut_input_lbns() = {kPackedBlobName};
  //
  Connect(faker_chain, chain_gph->NewEdge(), save_chain);
  chain_gph->UpdateSourceAndSink();
  chain_gph->ToDotWithAutoFilePath();
  BuildFromChainGph<MdSaveCompTaskNode>(std::move(chain_gph), false);
  ForEachNode([this](TaskNode* node) {
    auto model_save_comp_task_node = dynamic_cast<MdSaveCompTaskNode*>(node);
    if (model_save_comp_task_node != nullptr) {
      auto model_update_comp_task_node =
          static_cast<MdUpdtCompTaskNode*>(update_task_);
      model_save_comp_task_node->set_fw_task(
          model_update_comp_task_node->fw_task());
    }
  });
}

}  // namespace oneflow
