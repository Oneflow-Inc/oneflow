#include "graph/task_graph_manager.h"

namespace oneflow {

void TaskGraphMgr::Init(const JobSysConf& job_sys_conf) {
  IDMgr::Singleton().Init(job_sys_conf.resource());
  data_task_gph_.reset(new DataTaskGraph(job_sys_conf.train_dlnet_conf(),
                                         job_sys_conf.strategy(),
                                         true));
  // model_update_task_graph
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
    auto md_updt_gph = make_unique<MdUpdtTaskGraph> (
        data_chain.get(),
        data_chain2sorted_bp_comp_tasks.at(data_chain.get()));
    auto md_load_gph = make_unique<MdLoadTaskGraph> (md_updt_gph.get());
    md_updt_task_gphs_.emplace(data_chain.get(), std::move(md_updt_gph));
  }
}

} // namespace oneflow
