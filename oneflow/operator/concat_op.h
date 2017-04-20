#ifndef ONEFLOW_OPERATOR_CONCAT_OP_H_
#define ONEFLOW_OPERATOR_CONCAT_OP_H_

#include "operator/operator.h"

namespace oneflow {

class ConcatOp final : public UserOperator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ConcatOp);
  ConcatOp() = default;
  ~ConcatOp() = default;

  void Init(const OperatorConf& op_conf) override;

  std::string normal_ibn2lbn(const std::string& input_bn) const override { TODO(); }

  void InferShape4ObAndDtbFromIb() const override { TODO(); }
  void InferShape4Mtb() const override { TODO(); }
  void InferShape4Mdb() const override { TODO(); }

 private:
};

} // namespace oneflow

#endif // ONEFLOW_OPERATOR_CONCAT_OP_H_
