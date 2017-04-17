#include "graph/task_graph_manager.h"

namespace oneflow {

void TaskGraphMgr::Init(const JobSysConf& job_sys_conf) {
  IDMgr::Singleton().Init(job_sys_conf.resource());
  data_task_gph_.reset(new DataTaskGraph(job_sys_conf.train_dlnet_conf(),
                                         job_sys_conf.strategy(),
                                         true));
  // model_update_task_graph
  ChainAsKeyMap<std::vector<CompTaskNode*>> chain2sorted_bp_comp_tasks;
  for (const auto& node : data_task_gph_->nodes()) {
    auto bp_node = dynamic_cast<CompTaskNode*>(node.get());
    if (!bp_node || bp_node->IsFwNode()) { continue; }
    chain2sorted_bp_comp_tasks[bp_node->chain_node()].push_back(bp_node);
  }
  for (auto& pair : chain2sorted_bp_comp_tasks) {
    SortByParallelId(&(pair.second));
  }
  for (const auto& chain : data_task_gph_->chain_gph()->nodes()) {
    std::unique_ptr<MdUpdtTaskGraph> gph(
        new MdUpdtTaskGraph(chain.get(),
                            chain2sorted_bp_comp_tasks.at(chain.get())));
    md_updt_task_gphs_.insert(std::make_pair(chain.get(), std::move(gph)));
  }
}

} // namespace oneflow
