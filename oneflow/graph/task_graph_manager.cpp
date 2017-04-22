#include "graph/task_graph_manager.h"

namespace oneflow {

void TaskGraphMgr::Init() {
  LOG(INFO) << "Build Data";
  data_task_gph_.reset(new DataTaskGraph(
        JobDesc::Singleton().train_dlnet_conf(),
        JobDesc::Singleton().strategy(),
        true));
  // model_graph
  ChainAsKeyMap<std::vector<CompTaskNode*>> data_chain2sorted_bp_comp_tasks;
  for (const auto& node : data_task_gph_->nodes()) {
    auto bp_node = dynamic_cast<CompTaskNode*>(node.get());
    if (!bp_node || bp_node->IsFwNode()) { continue; }
    data_chain2sorted_bp_comp_tasks[bp_node->chain_node()].push_back(bp_node);
  }
  for (auto& pair : data_chain2sorted_bp_comp_tasks) {
    SortByParallelId(&(pair.second));
  }
  for (const auto& data_chain : data_task_gph_->chain_gph()->nodes()) {
    // model update
    auto md_updt_gph = of_make_unique<MdUpdtTaskGraph> (
        data_chain.get(),
        data_chain2sorted_bp_comp_tasks.at(data_chain.get()));
    ChainNode* updt_chain = md_updt_gph->chain_gph()->SoleLastNode();
    auto sorted_updt_tasks = md_updt_gph->SortedTasksInChain(updt_chain);
    // model load save
    auto md_load_gph =
        of_make_unique<MdLoadTaskGraph> (updt_chain, sorted_updt_tasks);
    auto md_save_gph =
        of_make_unique<MdSaveTaskGraph> (updt_chain, sorted_updt_tasks);
  }
}

} // namespace oneflow
