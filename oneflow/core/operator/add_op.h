#ifndef ONEFLOW_CORE_OPERATOR_ADD_OP_H_
#define ONEFLOW_CORE_OPERATOR_ADD_OP_H_

#include "oneflow/core/operator/cwise_op.h"

namespace oneflow {

class AddOp final : public CWiseOp {
 public:
  OF_DISALLOW_COPY_AND_MOVE(AddOp);
  AddOp() = default;
  ~AddOp() = default;

  void VirtualInitFromOpConf() override;
  const PbMessage& GetCustomizedConf() const override;

 private:
  Maybe<void> InferBatchAxis(
      std::function<OptInt64*(const std::string&)> BatchAxis4BnInOp) const override;

  void GetSbpSignatures(
      const std::function<const BlobDesc&(const std::string&)>& LogicalBlobDesc4Ibn,
      SbpSignatureList* sbp_sig_list) const override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_ADD_OP_H_
