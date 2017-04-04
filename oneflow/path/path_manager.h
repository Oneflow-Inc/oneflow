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
  ~PathManager() = default;

  static PathManager& Singleton() {
    static PathManager obj;
    return obj;
  }

  void Init(const JobSysConf& job_sys_conf);

 private:
  PathManager() = default;

  std::unique_ptr<DataPath> data_path_;
  std::unordered_map<const ChainNode*, std::unique_ptr<ModelUpdatePath>> model_update_paths_;
  std::unordered_map<const ChainNode*, std::unique_ptr<ModelLoadPath>> model_load_paths_;
  std::unordered_map<const ChainNode*, std::unique_ptr<ModelSavePath>> model_save_paths_;

};

} // namespace oneflow

#endif // ONEFLOW_PATH_PATH_MANAGER_H_
