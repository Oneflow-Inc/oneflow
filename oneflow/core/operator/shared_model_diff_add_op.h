#ifndef ONEFLOW_CORE_OPERATOR_SHARED_MODEL_DIFF_ADD_OP_H_
#define ONEFLOW_CORE_OPERATOR_SHARED_MODEL_DIFF_ADD_OP_H_

#include "oneflow/core/operator/add_op.h"

namespace oneflow {

class SharedModelDiffAddOp final : public AddOp {
 public:
  OF_DISALLOW_COPY_AND_MOVE(SharedModelDiffAddOp);
  SharedModelDiffAddOp() = default;
  ~SharedModelDiffAddOp() = default;

  void VirtualInitFromOpConf() override;
  const PbMessage& GetCustomizedConf() const override;

 private:
  LogicalBlobId ibn2lbi(const std::string& input_bn) const override { return GenPackedLbi(); }
  LogicalBlobId obn2lbi(const std::string& output_bn) const override { return GenPackedLbi(); }
};
}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_SHARED_MODEL_DIFF_ADD_OP_H_
