#ifndef OPERATOR_LOADER_OP_H_
#define OPERATOR_LOADER_OP_H_

#include "operator/operator.h"

namespace oneflow {

class LoaderOp final : public Operator {
 public:
  DISALLOW_COPY_AND_MOVE(LoaderOp);
  LoaderOp() = default;
  ~LoaderOp() = default;
  
  void Init(const OperatorConf& op_conf) override;
  bool IsElemWise() const override { return false; }

 private:

};

} // namespace oneflow

#endif // OPERATOR_LOADER_OP_H_
