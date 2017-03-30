#include "path/path_manager.h"

namespace oneflow {

void PathManager::Init(const JobSysConf& job_sys_conf) {
  IDManager::Singleton().Init(job_sys_conf.resource());
  // build data path
  std::unique_ptr<DataPath> data_path(new DataPath);
  data_path->Build(job_sys_conf.train_dlnet_conf(),
                   job_sys_conf.strategy(),
                   true); // TODO
  paths_.insert(std::make_pair("data", std::move(data_path)));
  // build model path
  std::vector<CpsDesc> cps_desc_vec;
  auto add_cps_desc = [&cps_desc_vec](const CpsDesc& cps_desc) {
    cps_desc_vec.push_back(cps_desc);
  };
  for (const auto& chain : paths_.at("data")->chain_graph()->nodes()) {
    std::unique_ptr<ModelUpdatePath> model_update_path(new ModelUpdatePath);
    std::unique_ptr<ModelLoadPath> model_load_path(new ModelLoadPath);
    std::unique_ptr<ModelSavePath> model_save_path(new ModelSavePath);
    model_update_path->Build(chain.get(), add_cps_desc);
    model_load_path->Build(chain.get(), add_cps_desc);
    model_save_path->Build(chain.get(), add_cps_desc);
    // TODO: name
    paths_.insert(std::make_pair("", std::move(model_update_path)));
    paths_.insert(std::make_pair("", std::move(model_load_path)));
    paths_.insert(std::make_pair("", std::move(model_save_path)));
  }
  // processs cross path subscribe
  for (const CpsDesc& cps_desc : cps_desc_vec) {
    ProcessCps(cps_desc);
  }
}

} // namespace oneflow
