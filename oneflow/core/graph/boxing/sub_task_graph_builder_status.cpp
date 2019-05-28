#include "oneflow/core/graph/boxing/sub_task_graph_builder_status.h"

namespace oneflow {

SubTskGphBuilderStatus::SubTskGphBuilderStatus(bool ok, std::string msg)
    : ok_(ok), msg_(std::move(msg)) {}

bool SubTskGphBuilderStatus::ok() const { return ok_; }

const std::string& SubTskGphBuilderStatus::msg() const { return msg_; }

SubTskGphBuilderStatus SubTskGphBuilderStatus::MakeStatusOK() {
  return SubTskGphBuilderStatus(true, "");
}

SubTskGphBuilderStatus SubTskGphBuilderStatus::MakeStatusOK(std::string msg) {
  return SubTskGphBuilderStatus(true, std::move(msg));
}

SubTskGphBuilderStatus SubTskGphBuilderStatus::MakeStatusError() {
  return SubTskGphBuilderStatus(false, "");
}

SubTskGphBuilderStatus SubTskGphBuilderStatus::MakeStatusError(std::string msg) {
  return SubTskGphBuilderStatus(false, std::move(msg));
}

}  // namespace oneflow
