#ifndef ONEFLOW_CORE_JOB_COMPLETER_AUTO_MIXED_PRECISION_LISTS_H_
#define ONEFLOW_CORE_JOB_COMPLETER_AUTO_MIXED_PRECISION_LISTS_H_

#include "oneflow/core/common/util.h"
#include "oneflow/core/operator/op_conf_util.h"

namespace oneflow {

typedef HashSet<std::string> AMPList;

class AutoMixedPrecisionLists final {
 public:
  // TODO(niuchong): list include grad
  static const AMPList& WhiteList();
  static const AMPList& BlackList();
  static const AMPList& GrayList();
  static const AMPList& ClearList();
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_JOB_COMPLETER_AUTO_MIXED_PRECISION_LISTS_H_
