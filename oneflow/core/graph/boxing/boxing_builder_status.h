#ifndef ONEFLOW_CORE_GRAPH_BOXING_BOXING_BUILDER_STATUS_H_
#define ONEFLOW_CORE_GRAPH_BOXING_BOXING_BUILDER_STATUS_H_

#include "oneflow/core/common/util.h"

namespace oneflow {

class BoxingBuilderStatus {
 public:
  BoxingBuilderStatus(bool ok, std::string msg);
  ~BoxingBuilderStatus() = default;

  bool ok() const;
  const std::string& msg() const;

  static BoxingBuilderStatus MakeStatusOK();
  static BoxingBuilderStatus MakeStatusOK(std::string msg);
  static BoxingBuilderStatus MakeStatusError();
  static BoxingBuilderStatus MakeStatusError(std::string msg);

 private:
  bool ok_;
  std::string msg_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_GRAPH_BOXING_BOXING_BUILDER_STATUS_H_
