#ifndef ONEFLOW_CORE_OPERATOR_MODEL_UPDATE_OP_H_
#define ONEFLOW_CORE_OPERATOR_MODEL_UPDATE_OP_H_

#include "oneflow/core/operator/operator.h"

namespace oneflow {

class ModelUpdtOp : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ModelUpdtOp);
  virtual ~ModelUpdtOp() = default;

  void InitFromOpConf() override {
    EnrollInputBn("model_diff_acc");
    EnrollOutputBn("model");
    if (JobDesc::Singleton()->regularization_method() != kNone) {
      EnrollDataTmpBn("regularized_diff");
    }
    VirtualInitFromOpConf();
  }

  void InferBlobDescs(
      std::function<BlobDesc*(const std::string)> GetBlobDesc4BnInOp,
      const ParallelContext* parallel_ctx) const {
    if (JobDesc::Singleton()->regularization_method() == kNone) { return; }
    const BlobDesc* model_blob_desc = GetBlobDesc4BnInOp("model");
    CHECK_EQ(model_blob_desc->data_type(),
             JobDesc::Singleton()->DefaultDataType());
    CHECK_EQ(model_blob_desc->has_data_id(), false);
    *GetBlobDesc4BnInOp("regularized_diff") = *model_blob_desc;
    VirtualInferBlobDescs(GetBlobDesc4BnInOp, parallel_ctx);
  }

 protected:
  ModelUpdtOp() = default;
  virtual void VirtualInitFromOpConf(){};
  virtual void VirtualInferBlobDescs(
      std::function<BlobDesc*(const std::string)> GetBlobDesc4BnInOp,
      const ParallelContext* parallel_ctx) const {};

 private:
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_MODEL_UPDATE_OP_H_
