#ifndef ONEFLOW_CORE_OPERATOR_ACCUMULATE_OP_H_
#define ONEFLOW_CORE_OPERATOR_ACCUMULATE_OP_H_

#include "oneflow/core/operator/operator_manager.h"

namespace oneflow {

class AccumulateOp final : public SysOperator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(AccumulateOp);
  AccumulateOp() = default;
  ~AccumulateOp() = default;

  void InitFromOpConf(const OperatorConf& op_conf) override;
  const PbMessage& GetSpecialConf() const override;

 private:
  std::string ibn2lbn(const std::string& input_bn) const override {
    return kPackedBlobName;
  }
  std::string obn2lbn(const std::string& output_bn) const override {
    return kPackedBlobName;
  }
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_ACCUMULATE_OP_H_
