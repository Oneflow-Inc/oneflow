#ifndef ONEFLOW_CORE_JOB_COMPLETER_AUTO_MIXED_PRECISION_LISTS_H_
#define ONEFLOW_CORE_JOB_COMPLETER_AUTO_MIXED_PRECISION_LISTS_H_

#include "oneflow/core/common/util.h"
#include "oneflow/core/operator/op_conf.pb.h"

namespace oneflow {

struct OpTypeCaseHash {
  std::size_t operator()(const OperatorConf::OpTypeCase& a) const {
    return std::hash<int>()(static_cast<int>(a));
  }
};

class AutoMixedPrecisionLists final {
 public:
  // TODO(niuchong): list include grad
  static const HashSet<OperatorConf::OpTypeCase, OpTypeCaseHash>& WhiteList();
  static const HashSet<OperatorConf::OpTypeCase, OpTypeCaseHash>& BlackList();
  static const HashSet<OperatorConf::OpTypeCase, OpTypeCaseHash>& GrayList();
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_JOB_COMPLETER_AUTO_MIXED_PRECISION_LISTS_H_
