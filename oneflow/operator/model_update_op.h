#ifndef ONEFLOW_OPERATOR_MODEL_UPDATE_OP_H_
#define ONEFLOW_OPERATOR_MODEL_UPDATE_OP_H_

#include "operator/operator.h"
#include "graph/register_desc.h"
namespace oneflow {

class ModelUpdateOp final : public SysOperator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ModelUpdateOp);
  ModelUpdateOp() = default;
  ~ModelUpdateOp() = default;

  std::string GetValueFromPbOpConf(const std::string& k) const override;
  void InitFromOpConf(const OperatorConf& op_conf) override;

  std::string normal_ibn2lbn(const std::string& input_bn) const override {
    return RegstDesc::kAllLbn;
  }
  std::string obn2lbn(const std::string& output_bn) const override {
    return RegstDesc::kAllLbn;
  }

  void InferShape4ObAndDtbFromIb() const override { UNEXPECTED_RUN(); }

 private:
};

} // namespace oneflow

#endif // ONEFLOW_OPERATOR_MODEL_UPDATE_OP_H_
