#ifndef ONEFLOW_PATH_MODEL_SAVE_PATH_H_
#define ONEFLOW_PATH_MODEL_SAVE_PATH_H_

#include "path/model_path.h"

namespace oneflow {

class ModelSavePath final : public ModelPath {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ModelSavePath);
  ModelSavePath() = default;
  ~ModelSavePath() = default;
  
  CompTaskNodeMemFunc Func4FwBuildExecAndProducedRegisters() const override {
    return &CompTaskNode::ModelSaveFwBuildExecAndProducedRegisters;
  }

  void Build(const ChainNode* chain_in_data_path) {
    LOG(FATAL) << "TODO";
  }

 private:
};

} // namespace oneflow

#endif // ONEFLOW_PATH_MODEL_SAVE_PATH_H_
