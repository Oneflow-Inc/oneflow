#ifndef ONEFLOW_OPERATOR_OPERATOR_FACTORY_H_
#define ONEFLOW_OPERATOR_OPERATOR_FACTORY_H_

#include "operator/operator.h"
#include "operator/op_conf.pb.h"

namespace oneflow {

class OperatorFactory final {
 public:
  DISALLOW_COPY_AND_MOVE(OperatorFactory);
  ~OperatorFactory() = default;
  static const OperatorFactory& singleton() {
    static OperatorFactory obj;
    return obj;
  }
  
  std::unique_ptr<Operator> ConstructOp(const OperatorConf&) const;

 private:
  OperatorFactory() = default;

};

} // namespace oneflow

#endif // ONEFLOW_OPERATOR_OPERATOR_FACTORY_H_
