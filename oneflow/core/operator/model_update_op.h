#ifndef ONEFLOW_CORE_OPERATOR_MODEL_UPDATE_OP_H_
#define ONEFLOW_CORE_OPERATOR_MODEL_UPDATE_OP_H_

#include "oneflow/core/operator/operator.h"

namespace oneflow {

class ModelUpdtOp : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ModelUpdtOp);
  virtual ~ModelUpdtOp() = default;

  virtual void InitFromOpConf() {
     EnrollInputBn("model_diff_acc");
     EnrollOutputBn("model");
     if (JobDesc::Singleton()->regularization_method != kNone) {
        EnrollDataTmp("regularized_diff");
     }
  }

  virtual void InferBlobDescs(
      std::function<BlobDesc*(const std::string)> GetBlobDesc4BnInOp,
      const ParallelContext* parallel_ctx) {
     if (JobDesc::Singleton()->regularization_method == kNone) { return; }
     const BlobDesc* model_blob_desc = GetBlobDesc4BnInOp("model");
     CHECK_EQ(model_blob_desc->data_type(),
              JobDesc::Singleton()->DefaultDataType());
     CHECK_EQ(model_blob_desc->has_data_id(), false);
     *GetBlobDesc4BnInOp("regularized__diff") = *model_blob_desc;
  }

 protected:
  ModelUpdtOp() = default;

 private:
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_MODEL_UPDATE_OP_H_
