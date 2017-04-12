#ifndef ONEFLOW_PATH_PATH_MANAGER_H_
#define ONEFLOW_PATH_PATH_MANAGER_H_

#include "job/job_conf.pb.h"
#include "path/data_path.h"
#include "path/model_load_path.h"
#include "path/model_save_path.h"
#include "path/model_update_path.h"

namespace oneflow {

class PathMgr {
 public:
  OF_DISALLOW_COPY_AND_MOVE(PathMgr);
  ~PathMgr() = default;

  static PathMgr& Singleton() {
    static PathMgr obj;
    return obj;
  }

  void Init(const JobSysConf& job_sys_conf);

 private:
  template<typename ValType>
  using ChainAsKeyMap = HashMap<const ChainNode*, ValType>;

  PathMgr() = default;

  std::unique_ptr<DataPath> data_path_;

  ChainAsKeyMap<std::unique_ptr<ModelUpdatePath>> model_update_paths_;
  ChainAsKeyMap<std::unique_ptr<ModelLoadPath>> model_load_paths_;
  ChainAsKeyMap<std::unique_ptr<ModelSavePath>> model_save_paths_;

};

} // namespace oneflow

#endif // ONEFLOW_PATH_PATH_MANAGER_H_
