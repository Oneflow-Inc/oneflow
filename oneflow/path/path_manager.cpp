#include "path/path_manager.h"

namespace oneflow {

void PathManager::Init(const JobSysConf& job_sys_conf) {
  IDManager::Singleton().Init(job_sys_conf.resource());
  // build data path
  data_path_.reset(new DataPath);
  data_path_->Build(job_sys_conf.train_dlnet_conf(),
                    job_sys_conf.strategy(),
                    true);
  // init data_chain2sorted_bp_comp_tasks
  ChainAsKeyMap<std::vector<CompTaskNode*>> data_chain2sorted_bp_comp_tasks;
  for (const auto& node : data_path_->task_gph()->nodes()) {
    auto bp_node = dynamic_cast<CompTaskNode*>(node.get());
    if (!bp_node || bp_node->IsFwNode()) { continue; }
    data_chain2sorted_bp_comp_tasks[bp_node->chain_node()].push_back(bp_node);
  }
  for (auto& pair : data_chain2sorted_bp_comp_tasks) {
    SortByParallelId(&(pair.second));
  }
  // build model path
  for (const auto& chain_uptr : data_path_->chain_gph()->nodes()) {
    const ChainNode* chain = chain_uptr.get();
    std::unique_ptr<ModelUpdatePath> update_path(new ModelUpdatePath);
    std::unique_ptr<ModelLoadPath> load_path(new ModelLoadPath);
    std::unique_ptr<ModelSavePath> save_path(new ModelSavePath);
    update_path->Build(chain, data_chain2sorted_bp_comp_tasks.at(chain));
    load_path->Build(chain);
    save_path->Build(chain);
    update_paths_.insert(std::make_pair(chain.get(), std::move(update_path)));
    load_paths_.insert(std::make_pair(chain.get(), std::move(load_path)));
    save_paths_.insert(std::make_pair(chain.get(), std::move(save_path)));
  }
}

} // namespace oneflow
