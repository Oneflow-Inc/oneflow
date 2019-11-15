#ifndef ONEFLOW_CORE_OPERATOR_OP_CONF_UTIL_H_
#define ONEFLOW_CORE_OPERATOR_OP_CONF_UTIL_H_

#include "oneflow/core/operator/op_conf.pb.h"

namespace std {

template<>
struct hash<::oneflow::OperatorConf::OpTypeCase> {
  std::size_t operator()(const ::oneflow::OperatorConf::OpTypeCase& op_type) const {
    return std::hash<int>()(static_cast<size_t>(op_type));
  }
};

}  // namespace std

#endif  // ONEFLOW_CORE_OPERATOR_OP_CONF_UTIL_H_
