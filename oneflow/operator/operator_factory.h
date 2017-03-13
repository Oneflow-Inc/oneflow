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
  
  std::shared_ptr<const Operator> ConstructOp(const OperatorConf&) const;

 private:
  OperatorFactory() = default;

};

inline std::shared_ptr<const Operator> ConstructOpFromPbConf(
    const OperatorConf pb_conf) {
  return OperatorFactory::singleton().ConstructOp(pb_conf);
}

} // namespace oneflow

#endif // ONEFLOW_OPERATOR_OPERATOR_FACTORY_H_
