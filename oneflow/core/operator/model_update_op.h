#ifndef ONEFLOW_CORE_OPERATOR_MODEL_UPDATE_OP_H_
#define ONEFLOW_CORE_OPERATOR_MODEL_UPDATE_OP_H_

#include "oneflow/core/operator/operator_manager.h"

namespace oneflow {

class ModelUpdtOp : public SysOperator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ModelUpdtOp);
  virtual ~ModelUpdtOp() = default;

  virtual void InferShape4FwBlobs(
      std::function<Shape*(const std::string&)> GetShapePtr4BnInOp,
      ParallelPolicy policy, int64_t parallel_id,
      int64_t parallel_num) const override {
    // do nothing
  }

 protected:
  ModelUpdtOp() = default;

 private:
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_MODEL_UPDATE_OP_H_
