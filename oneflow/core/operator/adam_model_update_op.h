#ifndef ONEFLOW_CORE_OPERATOR_ADAM_MODEL_UPDATE_OP_H_
#define ONEFLOW_CORE_OPERATOR_ADAM_MODEL_UPDATE_OP_H_

#include "oneflow/core/operator/normal_model_update_op.h"

namespace oneflow {

class AdamModelUpdateOp final : public NormalModelUpdtOp {
 public:
  OF_DISALLOW_COPY_AND_MOVE(AdamModelUpdateOp);
  AdamModelUpdateOp() = default;
  ~AdamModelUpdateOp() = default;

 private:
  void MdUpdtVirtualInitFromOpConf() override;
  void MdUpdtVirtualInferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                                   const ParallelContext* parallel_ctx) const override;
  void InferInputBlobParallelDesc(
      std::function<BlobParallelDesc*(const std::string&)> BlobParallelDesc4BnInOp,
      const ParallelContext* parallel_context) const override {
    UNIMPLEMENTED();
  }
  void InferOutputAndDataTmpBlobParallelDesc(
      std::function<BlobParallelDesc*(const std::string&)> BlobParallelDesc4BnInOp,
      const ParallelContext* parallel_context) const override {
    UNIMPLEMENTED();
  }
  void InferModelBlobParallelDesc(
      std::function<BlobParallelDesc*(const std::string&)> BlobParallelDesc4BnInOp,
      const ParallelContext* parallel_context) const override {
    UNIMPLEMENTED();
  }
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_ADAM_MODEL_UPDATE_OP_H_
