#include "graph/task_graph_manager.h"

namespace oneflow {

void TaskGraphMgr::Init() {
  // data graph
  task_gphs_.clear();
  LOG(INFO) << "Build Data";
  auto data_task_gph = new DataTaskGraph(
        JobDesc::Singleton().train_dlnet_conf(),
        JobDesc::Singleton().strategy(),
        true);
  task_gphs_.emplace_back(data_task_gph);
  // construct data_chain2sorted_bp_comp_tasks
  HashMap<const ChainNode*, std::vector<CompTaskNode*>>
      data_chain2sorted_bp_comp_tasks;
  for (const auto& node : data_task_gph->nodes()) {
    auto bp_node = dynamic_cast<CompTaskNode*>(node.get());
    if (bp_node == nullptr || bp_node->IsFwNode()) { continue; }
    data_chain2sorted_bp_comp_tasks[bp_node->chain_node()].push_back(bp_node);
  }
  for (auto& pair : data_chain2sorted_bp_comp_tasks) {
    SortByParallelId(&(pair.second));
  }
  // model graph
  for (const auto& data_chain : data_task_gph->chain_gph()->nodes()) {
    // model update
    auto md_updt_gph = new MdUpdtTaskGraph(
        data_chain.get(),
        data_chain2sorted_bp_comp_tasks.at(data_chain.get()));
    ChainNode* updt_chain = md_updt_gph->chain_gph()->SoleLastNode();
    auto sorted_updt_tasks = md_updt_gph->SortedCompTasksInChain(updt_chain);
    // model load save
    auto md_load_gph = new MdLoadTaskGraph(updt_chain, sorted_updt_tasks);
    auto md_save_gph = new MdSaveTaskGraph(updt_chain, sorted_updt_tasks);
    task_gphs_.emplace_back(md_updt_gph);
    task_gphs_.emplace_back(md_load_gph);
    task_gphs_.emplace_back(md_save_gph);
  }
}

} // namespace oneflow
