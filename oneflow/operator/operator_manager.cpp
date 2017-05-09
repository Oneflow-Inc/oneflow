#include "operator/operator_manager.h"
#include "glog/logging.h"

namespace oneflow {

HashMap<int, std::function<std::shared_ptr<Operator>()>>&
OpMgr::OpTypeCase2Creator() {
  static HashMap<int, std::function<std::shared_ptr<Operator>()>> obj;
  return obj;
}

std::shared_ptr<Operator> OpMgr::ConstructOp(
    const OperatorConf& op_conf) {
  auto ret = OpTypeCase2Creator().at(op_conf.op_type_case())();
  ret->InitFromOpConf(op_conf);
  op_list_.emplace_back(ret);
  return ret;
}

std::shared_ptr<Operator> OpMgr::ConstructOp(
    const OperatorProto& op_proto) {
  auto ret = OpTypeCase2Creator().at(op_proto.op_conf().op_type_case())();
  ret->InitFromProto(op_proto);
  return ret;
}

PbVector<OperatorProto> OpMgr::ToProto4AllOp() {
  PbVector<OperatorProto> ret;
  for (auto it = op_list_.begin(); it != op_list_.end();) {
    if (std::shared_ptr<const Operator> op = it->lock()) {
      *(ret.Add()) = op->ToProto();
      ++it;
    } else {
      auto cur_it = it++;
      op_list_.erase(cur_it);
    }
  }
  return ret;
}

} // namespace oneflow
