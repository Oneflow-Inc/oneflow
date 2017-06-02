#ifndef ONEFLOW_OPERATOR_OPERATOR_MANAGER_H_
#define ONEFLOW_OPERATOR_OPERATOR_MANAGER_H_

#include <iostream>
#include <functional>
#include <list>
#include "oneflow/core/operator/operator.h"
#include "oneflow/core/operator/op_conf.pb.h"

namespace oneflow {

class OpMgr final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(OpMgr);
  ~OpMgr() = default;
  
  static OpMgr& Singleton() {
    static OpMgr obj;
    return obj;
  }
  
  std::shared_ptr<Operator> ConstructOp(const OperatorConf&);

  void AllOpToProto(PbRpf<OperatorProto>*);

 private:
  OpMgr() = default;

  std::list<std::weak_ptr<const Operator>> op_list_;

};

void AddOpCreator(OperatorConf::OpTypeCase op_type_case,
                  std::function<Operator*()> creator);

Operator* CreateOp(OperatorConf::OpTypeCase op_type_case);

template<OperatorConf::OpTypeCase op_type_case, typename OpType>
struct OpRegister {
  OpRegister() {
    AddOpCreator(op_type_case, []() { return new OpType; });
  }
};

#define REGISTER_OP(OpTypeCase, OpType) \
  static OpRegister<OpTypeCase, OpType> g_##OpType##_register_var;

} // namespace oneflow

#endif // ONEFLOW_OPERATOR_OPERATOR_MANAGER_H_
