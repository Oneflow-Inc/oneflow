#ifndef ONEFLOW_OPERATOR_CLONE_OP_H_
#define ONEFLOW_OPERATOR_CLONE_OP_H_

#include "operator/operator.h"

namespace oneflow {

class CloneOp final : public SysOperator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CloneOp);
  CloneOp() = default;
  ~CloneOp() = default;

  void Init(const OperatorConf& op_conf) override;
  void InferShape4ObAndDtbFromIb() const override { TODO(); }
  
  std::string normal_ibn2lbn(const std::string& input_bn) const override {
    return GetValueFromPbOpConf("lbn");
  }
  std::string obn2lbn(const std::string& output_bn) const override {
    if (is_boxing_) {
      return GetValueFromPbOpConf("lbn");
    } else {
      return op_name() + "/" + output_bn;
    }
  }

 private:
  bool is_boxing_;

};

} // namespace oneflow

#endif // ONEFLOW_OPERATOR_CLONE_OP_H_
