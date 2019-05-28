#ifndef ONEFLOW_CORE_GRAPH_BOXING_SUB_TASK_GRAPH_BUILDER_STATUS_H_
#define ONEFLOW_CORE_GRAPH_BOXING_SUB_TASK_GRAPH_BUILDER_STATUS_H_

#include "oneflow/core/common/util.h"

namespace oneflow {

class SubTskGphBuilderStatus {
 public:
  SubTskGphBuilderStatus(bool ok, std::string msg);
  ~SubTskGphBuilderStatus() = default;

  bool ok() const;
  const std::string& msg() const;

  static SubTskGphBuilderStatus MakeStatusOK();
  static SubTskGphBuilderStatus MakeStatusOK(std::string msg);
  static SubTskGphBuilderStatus MakeStatusError();
  static SubTskGphBuilderStatus MakeStatusError(std::string msg);

 private:
  bool ok_;
  std::string msg_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_GRAPH_BOXING_SUB_TASK_GRAPH_BUILDER_STATUS_H_
