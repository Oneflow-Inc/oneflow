#include "path/path_manager.h"

namespace oneflow {

void PathManager::Init(const JobSysConf& job_sys_conf) {
  // id map
  IDMap id_map;
  id_map.Init(job_sys_conf.resource());
  // build data path
  std::unique_ptr<DataPath> data_path(new DataPath);
  data_path->Build(job_sys_conf.train_dlnet_conf(),
                   job_sys_conf.strategy(),
                   id_map,
                   true);
  

  LOG(FATAL) << "TODO";
}

} // namespace oneflow
