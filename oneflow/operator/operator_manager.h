#ifndef ONEFLOW_OPERATOR_OPERATOR_MANAGER_H_
#define ONEFLOW_OPERATOR_OPERATOR_MANAGER_H_

#include <iostream>
#include <functional>
#include "operator/operator.h"
#include "operator/op_conf.pb.h"

namespace oneflow {

class OpMgr final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(OpMgr);
  ~OpMgr() = default;
  
  static OpMgr& Singleton() {
    static OpMgr obj;
    return obj;
  }
  
  std::shared_ptr<Operator> ConstructOp(const OperatorConf&) const;
  std::shared_ptr<Operator> ConstructOp(const OperatorProto&) const;

 private:
  template<OperatorConf::OpTypeCase op_type_case, typename OpType>
  friend struct OpRegister;

  OpMgr() = default;
  static HashMap<int, std::function<std::shared_ptr<Operator>()>>& OpTypeCase2Creator();

};

template<OperatorConf::OpTypeCase op_type_case, typename OpType>
struct OpRegister {
  OpRegister() {
    OpMgr::OpTypeCase2Creator().emplace(op_type_case, []() {
      return std::make_shared<OpType> ();;
    });
  }
};

#define REGISTER_OP(OpTypeCase, OpType) \
  static OpRegister<OpTypeCase, OpType> g_##OpType##_register_var;

} // namespace oneflow

#endif // ONEFLOW_OPERATOR_OPERATOR_MANAGER_H_
