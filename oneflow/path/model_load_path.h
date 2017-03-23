#ifndef ONEFLOW_PATH_MODEL_LOAD_PATH_H_
#define ONEFLOW_PATH_MODEL_LOAD_PATH_H_

#include "path/path.h"

namespace oneflow {

class ModelLoadPath final : public Path {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ModelLoadPath);
  ModelLoadPath() = default;
  ~ModelLoadPath() = default;

 private:
};

} // namespace oneflow

#endif // ONEFLOW_PATH_MODEL_LOAD_PATH_H_
