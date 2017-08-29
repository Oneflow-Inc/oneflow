#ifndef ONEFLOW_CORE_OPERATOR_OPERATOR_MANAGER_H_
#define ONEFLOW_CORE_OPERATOR_OPERATOR_MANAGER_H_

#include "oneflow/core/operator/op_conf.pb.h"
#include "oneflow/core/operator/operator.h"

namespace oneflow {

class OpMgr final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(OpMgr);
  ~OpMgr() = default;

  OF_SINGLETON(OpMgr);

  std::shared_ptr<Operator> AddOp(const OperatorConf&);

  void AllOpToProto(PbRpf<OperatorProto>*);

  std::shared_ptr<Operator> ModelUpdateOp();

 private:
  OpMgr() = default;

  std::list<std::weak_ptr<const Operator>> op_list_;
  std::shared_ptr<Operator> model_update_op_;
};

void AddOpCreator(OperatorConf::OpTypeCase op_type_case,
                  std::function<Operator*()> creator);

Operator* CreateOp(OperatorConf::OpTypeCase op_type_case);
std::shared_ptr<Operator> ConstructOp(const OperatorConf&);

template<OperatorConf::OpTypeCase op_type_case, typename OpType>
struct OpRegister {
  OpRegister() {
    AddOpCreator(op_type_case, []() { return new OpType; });
  }
};

#define REGISTER_OP(OpTypeCase, OpType) \
  static OpRegister<OpTypeCase, OpType> g_##OpType##_register_var;

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_OPERATOR_MANAGER_H_
