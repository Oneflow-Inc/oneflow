#ifndef ONEFLOW_CORE_OPERATOR_MODEL_DIFF_ACCUMULATE_OP_H_
#define ONEFLOW_CORE_OPERATOR_MODEL_DIFF_ACCUMULATE_OP_H_

#include "oneflow/core/operator/operator_manager.h"

namespace oneflow {

class ModelDiffAccOp final : public SysOperator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ModelDiffAccOp);
  ModelDiffAccOp() = default;
  ~ModelDiffAccOp() = default;

  void InitFromOpConf(const OperatorConf& op_conf) override;
  const PbMessage& GetSpecialConf() const override;

 private:
  std::string ibn2lbn(const std::string& input_bn) const override {
    return kBaledBlobName;
  }
  std::string obn2lbn(const std::string& output_bn) const override {
    return kBaledBlobName;
  }
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_MODEL_DIFF_ACCUMULATE_OP_H_
