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
  // void VirtualInferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
  //                           const ParallelContext* parallel_ctx) const override;
  virtual void RefineDiffBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                                   const ParallelContext*) const override;
};
}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_ADD_OP_H_
