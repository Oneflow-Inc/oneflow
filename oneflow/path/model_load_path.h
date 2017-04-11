#ifndef ONEFLOW_PATH_MODEL_LOAD_PATH_H_
#define ONEFLOW_PATH_MODEL_LOAD_PATH_H_

#include "path/model_path.h"

namespace oneflow {

class ModelLoadPath final : public ModelPath {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ModelLoadPath);
  ModelLoadPath() = default;
  ~ModelLoadPath() = default;
  
  CompTaskNodeMemFunc Func4FwBuildExecAndProducedRegsts() const override {
    return &CompTaskNode::ModelLoadFwBuildExecAndProducedRegsts;
  }

  void Build(const ChainNode* chain_in_data_path) {
    LOG(FATAL) << "TODO";
  }

 private:
};

} // namespace oneflow

#endif // ONEFLOW_PATH_MODEL_LOAD_PATH_H_
