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
  bool NeedInBlobWhenBackward() const override { return false; }
  bool NeedOutBlobWhenBackward() const override { return false; }
  bool CanInferInDiffDynamicShapeWithoutIn() const override { return true; }
  virtual void FixInDiffBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                                  const ParallelContext*) const override;
};
}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_ADD_OP_H_
