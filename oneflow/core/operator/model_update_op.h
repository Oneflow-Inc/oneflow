#ifndef ONEFLOW_CORE_OPERATOR_MODEL_UPDATE_OP_H_
#define ONEFLOW_CORE_OPERATOR_MODEL_UPDATE_OP_H_

#include "oneflow/core/operator/operator_manager.h"

namespace oneflow {

class ModelUpdtOp : public SysOperator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ModelUpdtOp);
  virtual ~ModelUpdtOp() = default;

  virtual void InferBlobDesc4FwBlobs(
      std::function<BlobDesc*(const std::string)> GetBlobDesc4BnInOp,
      ParallelPolicy policy, int64_t parallel_id,
      int64_t parallel_num) override {
    // do nothing
  }

 protected:
  ModelUpdtOp() = default;

 private:
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_MODEL_UPDATE_OP_H_
