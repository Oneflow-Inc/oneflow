#include "path/path_manager.h"

namespace oneflow {

void PathManager::Init(const JobSysConf& job_sys_conf) {
  IDManager::Singleton().Init(job_sys_conf.resource());
  // build data path
  data_path_.reset(new DataPath);
  data_path_->Build(job_sys_conf.train_dlnet_conf(),
                    job_sys_conf.strategy(),
                    true);
  // build model path
  for (const auto& chain : data_path_->chain_graph()->nodes()) {
    std::unique_ptr<ModelUpdatePath> model_update_path(new ModelUpdatePath);
    std::unique_ptr<ModelLoadPath> model_load_path(new ModelLoadPath);
    std::unique_ptr<ModelSavePath> model_save_path(new ModelSavePath);
    model_update_path->Build(chain.get());
    model_load_path->Build(chain.get());
    model_save_path->Build(chain.get());
    model_update_paths_.insert(std::make_pair(chain.get(), std::move(model_update_path)));
    model_load_paths_.insert(std::make_pair(chain.get(), std::move(model_load_path)));
    model_save_paths_.insert(std::make_pair(chain.get(), std::move(model_save_path)));
  }
}

} // namespace oneflow
