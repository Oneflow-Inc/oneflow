#ifndef ONEFLOW_CORE_OPERATOR_MODEL_UPDATE_OP_H_
#define ONEFLOW_CORE_OPERATOR_MODEL_UPDATE_OP_H_

#include "oneflow/core/operator/operator.h"

namespace oneflow {

class ModelUpdtOp : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ModelUpdtOp);
  virtual ~ModelUpdtOp() = default;

  virtual void InitFromOpConf() {
    EnrollInputBn("model_diff_acc", false);
    EnrollOutputBn("model", false);
    MdUpdtVirtualInitFromOpConf();
  }

  virtual void InferBlobDescs(
      std::function<BlobDesc*(const std::string)> GetBlobDesc4BnInOp,
      const ParallelContext* parallel_ctx) {}

 protected:
  ModelUpdtOp() = default;
  virtual void MdUpdtVirtualInitFromOpConf() {}

 private:
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_MODEL_UPDATE_OP_H_
