#include <utility>

#include "oneflow/core/graph/boxing/boxing_builder_status.h"

namespace oneflow {

BoxingBuilderStatus::BoxingBuilderStatus(bool ok, std::string msg)
    : ok_(ok), msg_(std::move(msg)) {}

bool BoxingBuilderStatus::ok() const { return ok_; }

const std::string& BoxingBuilderStatus::msg() const { return msg_; }

BoxingBuilderStatus BoxingBuilderStatus::MakeStatusOK() { return BoxingBuilderStatus(true, ""); }

BoxingBuilderStatus BoxingBuilderStatus::MakeStatusOK(std::string msg) {
  return BoxingBuilderStatus(true, std::move(msg));
}

BoxingBuilderStatus BoxingBuilderStatus::MakeStatusError() {
  return BoxingBuilderStatus(false, "");
}

BoxingBuilderStatus BoxingBuilderStatus::MakeStatusError(std::string msg) {
  return BoxingBuilderStatus(false, std::move(msg));
}

}  // namespace oneflow
