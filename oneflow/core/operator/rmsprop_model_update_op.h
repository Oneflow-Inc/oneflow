#ifndef ONEFLOW_CORE_OPERATOR_RMSPROP_MODEL_UPDATE_OP_H_
#define ONEFLOW_CORE_OPERATOR_RMSPROP_MODEL_UPDATE_OP_H_

#include "oneflow/core/operator/operator_manager.h"

namespace oneflow {

class RMSPropModelUpdateOp final : public SysOperator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(RMSPropModelUpdateOp);
  RMSPropModelUpdateOp() = default;
  ~RMSPropModelUpdateOp() = default;

  void InitFromOpConf(const OperatorConf& op_conf) override;
  const PbMessage& GetSpecialConf() const override;

 private:
  std::string ibn2lbn(const std::string& input_bn) const override {
    return kPackedBlobName;
  }
  std::string obn2lbn(const std::string& output_bn) const override {
    return kPackedBlobName;
  }
  std::string mtbn2lbn(const std::string& output_bn) const override {
    return kPackedBlobName;
  }
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_RMSPROP_MODEL_UPDATE_OP_H_
