#ifndef ONEFLOW_OPERATOR_OPERATOR_FACTORY_H_
#define ONEFLOW_OPERATOR_OPERATOR_FACTORY_H_

#include <iostream>
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
  std::shared_ptr<Operator> ConstructOp(const OperatorProto&) const;

 private:
  template<OperatorConf::OpTypeCase op_type_case, typename OpType>
  friend struct OpRegister;

  OpFactory() = default;
  static HashMap<int, std::function<std::shared_ptr<Operator>()>>& OpTypeCase2Creator();

};

template<OperatorConf::OpTypeCase op_type_case, typename OpType>
struct OpRegister {
  OpRegister() {
    OpFactory::OpTypeCase2Creator().emplace(op_type_case, []() {
      return std::make_shared<OpType> ();;
    });
  }
};

#define REGISTER_OP(OpTypeCase, OpType) \
  static OpRegister<OpTypeCase, OpType> g_##OpType##_register_var;

inline std::shared_ptr<Operator> ConstructOpFromPbConf(
    const OperatorConf& pb_conf) {
  return OpFactory::Singleton().ConstructOp(pb_conf);
}

} // namespace oneflow

#endif // ONEFLOW_OPERATOR_OPERATOR_FACTORY_H_
