#ifndef OPERATOR_INNERPRODUCT_OP_H_
#define OPERATOR_INNERPRODUCT_OP_H_

#include "operator/operator.h"

namespace oneflow {

class InnerProductOp final : public UserOperator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(InnerProductOp);
  InnerProductOp() = default;
  ~InnerProductOp() = default;

  void InitFromOpConf(const OperatorConf& op_conf) override;
  std::string GetValueFromPbOpConf(const std::string& k) const override;
  
  void InferShape4ObAndDtbFromIb() const override { TODO(); }
  void InferShape4Mtb() const override { TODO(); }
  void InferShape4Mdb() const override { TODO(); }

 private:

};

} // namespace oneflow

#endif // OPERATOR_INNERPRODUCT_OP_H_
