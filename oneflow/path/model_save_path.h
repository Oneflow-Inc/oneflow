#ifndef ONEFLOW_PATH_MODEL_SAVE_PATH_H_
#define ONEFLOW_PATH_MODEL_SAVE_PATH_H_

#include "path/path.h"

namespace oneflow {

class ModelSavePath final : public Path {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ModelSavePath);
  ModelSavePath() = default;
  ~ModelSavePath() = default;

  void Build(const ChainNode* chain_in_data_path,
             std::function<void(const CpsDesc&)> add_cps_desc) {
    LOG(FATAL) << "TODO";
  }

 private:
};

} // namespace oneflow

#endif // ONEFLOW_PATH_MODEL_SAVE_PATH_H_
