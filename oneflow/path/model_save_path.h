#ifndef ONEFLOW_PATH_MODEL_SAVE_PATH_H_
#define ONEFLOW_PATH_MODEL_SAVE_PATH_H_

#include "path/path.h"

namespace oneflow {

class ModelSavePath final : public Path {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ModelSavePath);
  ModelSavePath() = default;
  ~ModelSavePath() = default;

 private:
};

} // namespace oneflow

#endif // ONEFLOW_PATH_MODEL_SAVE_PATH_H_
