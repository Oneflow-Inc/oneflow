#ifndef ONEFLOW_CORE_OPERATOR_LOSS_RECORD_OP_H_
#define ONEFLOW_CORE_OPERATOR_LOSS_RECORD_OP_H_

#include "oneflow/core/operator/operator.h"

namespace oneflow {

class LossRecordOp final : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(LossRecordOp);
  LossRecordOp() = default;
  ~LossRecordOp() = default;

  void InitFromOpConf() override;
  const PbMessage& GetSpecialConf() const override;

 private:
  std::string ibn2lbn(const std::string& input_bn) const override {
    return kPackedBlobName;
  }
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_LOSS_RECORD_OP_H_
