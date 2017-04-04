#ifndef ONEFLOW_PATH_MODEL_UPDATE_PATH_H_
#define ONEFLOW_PATH_MODEL_UPDATE_PATH_H_

#include "path/path.h"

namespace oneflow {

class ModelUpdatePath final : public Path {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ModelUpdatePath);
  ModelUpdatePath() = default;
  ~ModelUpdatePath() = default;

  void Build(const ChainNode* chain_in_data_path) {
    LOG(FATAL) << "TODO";
  }

 private:
};

} // namespace oneflow

#endif // ONEFLOW_PATH_MODEL_UPDATE_PATH_H_
