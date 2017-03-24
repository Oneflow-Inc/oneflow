#ifndef ONEFLOW_PATH_DATA_PATH_H_
#define ONEFLOW_PATH_DATA_PATH_H_

#include "path/path.h"

namespace oneflow {

class DataPath final : public Path {
 public:
  OF_DISALLOW_COPY_AND_MOVE(DataPath);
  DataPath() = default;
  ~DataPath() = default;

  void Build(const DLNetConf& dl_net_conf,
             const Strategy& strategy_conf,
             const IDMap& id_map,
             bool need_bp) {
    mut_task_graph().reset(new TaskGraph);
    mut_task_graph()->Init(dl_net_conf, strategy_conf, id_map, need_bp);
  }

 private:
};

} // namespace oneflow

#endif // ONEFLOW_PATH_DATA_PATH_H_
