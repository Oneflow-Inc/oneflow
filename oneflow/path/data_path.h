#ifndef ONEFLOW_PATH_DATA_PATH_H_
#define ONEFLOW_PATH_DATA_PATH_H_

#include "path/path.h"

namespace oneflow {

class DataPath final : public Path {
 public:
  OF_DISALLOW_COPY_AND_MOVE(DataPath);
  DataPath() = default;
  ~DataPath() = default;
 
  CompTaskNode* Faker2Mccoy(CompTaskNode*) const override { UNEXPECTED_RUN(); }

  CompTaskNodeMemFunc Func4FwBuildExecAndProducedRegsts() const override {
    return &CompTaskNode::DataFwBuildExecAndProducedRegsts;
  }

  const ChainNode* GetDataChain() const override { UNEXPECTED_RUN(); }

  void Build(const DLNetConf& dl_net_conf,
             const Strategy& strategy_conf,
             bool need_bp);

 private:
};

void DataPath::Build(const DLNetConf& dl_net_conf,
                     const Strategy& strategy_conf,
                     bool need_bp) {
  mut_task_gph().reset(new TaskGraph(dl_net_conf, strategy_conf, need_bp));
  BuildExecAndProducedRegstsAndSubscribeInPath();
}

} // namespace oneflow

#endif // ONEFLOW_PATH_DATA_PATH_H_
