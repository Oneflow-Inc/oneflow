#include "operator/operator_factory.h"
#include <functional>
#include "glog/logging.h"

namespace oneflow {

HashMap<int, std::function<std::shared_ptr<Operator>()>>&
OpFactory::OpTypeCase2Creator() {
  static HashMap<int, std::function<std::shared_ptr<Operator>()>> obj;
  return obj;
}

std::shared_ptr<Operator> OpFactory::ConstructOp(
    const OperatorConf& op_conf) const {
  auto ret = OpTypeCase2Creator().at(op_conf.op_type_case())();
  ret->InitFromOpConf(op_conf);
  return ret;
}

std::shared_ptr<Operator> OpFactory::ConstructOp(
    const OperatorProto& op_proto) const {
  auto ret = OpTypeCase2Creator().at(op_proto.user_conf().op_type_case())();
  ret->InitFromOperatorProto(op_proto);
  return ret;
}

} // namespace oneflow
