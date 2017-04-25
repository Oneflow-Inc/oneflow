#ifndef ONEFLOW_OPERATOR_CLONE_OP_H_
#define ONEFLOW_OPERATOR_CLONE_OP_H_

#include "operator/operator.h"

namespace oneflow {

class CloneOp final : public SysOperator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CloneOp);
  CloneOp() = default;
  ~CloneOp() = default;
  
  bool IsElemWise() const override { return true; }

  void InitFromOpConf(const OperatorConf& op_conf) override;
  void InferShape4ObAndDtbFromIb() const override { TODO(); }
  std::string GetValueFromPbOpConf(const std::string& k) const override;
  
  std::string normal_ibn2lbn(const std::string& input_bn) const override {
    return GetValueFromPbOpConf("lbn");
  }
  std::string obn2lbn(const std::string& output_bn) const override {
    return op_name() + "/" + output_bn;
  }

 private:

};

} // namespace oneflow

#endif // ONEFLOW_OPERATOR_CLONE_OP_H_
