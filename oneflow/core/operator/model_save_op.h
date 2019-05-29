#ifndef ONEFLOW_CORE_OPERATOR_MODEL_SAVE_OP_H_
#define ONEFLOW_CORE_OPERATOR_MODEL_SAVE_OP_H_

#include "oneflow/core/operator/operator.h"

namespace oneflow {

class ModelSaveOp final : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ModelSaveOp);
  ModelSaveOp() = default;
  ~ModelSaveOp() = default;

  void InitFromOpConf() override;
  const PbMessage& GetCustomizedConf() const override;

 private:
  void InferHasBatchDim(
      std::function<bool*(const std::string&)> HasBatchDim4BnInOp) const override {}
  void GetSbpSignatures(
      const std::function<const BlobDesc&(const std::string&)>& LogicalBlobDesc4Ibn,
      SbpSignatureList* sbp_sig_list) const override {}
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_MODEL_SAVE_OP_H_
