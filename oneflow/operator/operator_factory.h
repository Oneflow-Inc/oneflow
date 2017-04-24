#ifndef ONEFLOW_OPERATOR_OPERATOR_FACTORY_H_
#define ONEFLOW_OPERATOR_OPERATOR_FACTORY_H_

#include "operator/operator.h"
#include "operator/op_conf.pb.h"

namespace oneflow {

class OpFactory final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(OpFactory);
  ~OpFactory() = default;
  static OpFactory& Singleton() {
    static OpFactory obj;
    return obj;
  }
  
  std::shared_ptr<Operator> ConstructOp(const OperatorConf&) const;

 private:
  OpFactory() = default;

};

inline std::shared_ptr<Operator> ConstructOpFromPbConf(
    const OperatorConf& pb_conf) {
  return OpFactory::Singleton().ConstructOp(pb_conf);
}

} // namespace oneflow

#endif // ONEFLOW_OPERATOR_OPERATOR_FACTORY_H_
