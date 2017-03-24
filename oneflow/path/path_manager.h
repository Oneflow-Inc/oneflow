#ifndef ONEFLOW_PATH_PATH_MANAGER_H_
#define ONEFLOW_PATH_PATH_MANAGER_H_

#include "job/job_conf.pb.h"
#include "path/data_path.h"
#include "path/model_load_path.h"
#include "path/model_save_path.h"
#include "path/model_update_path.h"

namespace oneflow {

class PathManager {
 public:
  OF_DISALLOW_COPY_AND_MOVE(PathManager);
  PathManager() = default;
  ~PathManager() = default;

  void Init(const JobSysConf& job_sys_conf);

 private:
  void ProcessCps(const CpsDesc& cps_desc) {
    LOG(FATAL) << "TODO";
  }

  std::unordered_map<std::string, std::unique_ptr<Path>> paths_;

};

} // namespace oneflow

#endif // ONEFLOW_PATH_PATH_MANAGER_H_
