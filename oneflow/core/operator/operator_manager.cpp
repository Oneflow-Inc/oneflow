#include "oneflow/core/operator/operator_manager.h"

namespace oneflow {

namespace {

HashMap<int, std::function<Operator*()>>& OpTypeCase2Creator() {
  static HashMap<int, std::function<Operator*()>> obj;
  return obj;
}

}

void AddOpCreator(OperatorConf::OpTypeCase op_type_case,
                  std::function<Operator*()> creator) {
  CHECK(OpTypeCase2Creator().emplace(op_type_case, creator).second);
}

Operator* CreateOp(OperatorConf::OpTypeCase op_type_case) {
  return OpTypeCase2Creator().at(op_type_case)();
}

std::shared_ptr<Operator> OpMgr::ConstructOp(
    const OperatorConf& op_conf) {
  std::shared_ptr<Operator> ret(CreateOp(op_conf.op_type_case()));
  ret->InitFromOpConf(op_conf);
  op_list_.emplace_back(ret);
  return ret;
}

void OpMgr::AllOpToProto(PbRpf<OperatorProto>* ret) {
  ret->Clear();
  for (auto it = op_list_.begin(); it != op_list_.end();) {
    if (std::shared_ptr<const Operator> op = it->lock()) {
      op->ToProto(ret->Add());
      ++it;
    } else {
      op_list_.erase(it++);
    }
  }
}

} // namespace oneflow
