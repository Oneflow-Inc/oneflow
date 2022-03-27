#include <memory>
#include <tuple>
#include <vector>
#include "oneflow/core/operator/variable_tensor_mgr.h"
#include "oneflow/core/common/just.h"
#include "oneflow/core/common/throw.h"
#include "oneflow/core/framework/tensor.h"

namespace oneflow {

Maybe<void> VariableTensorMgr::Set(const std::string& variable_op_name,
                                   const std::shared_ptr<one::Tensor>& variable_tensor) {
  variables_[variable_op_name] = variable_tensor;
  return Maybe<void>::Ok();
}

Maybe<one::Tensor> VariableTensorMgr::Get(const std::string& variable_op_name) {
  if (variables_.find(variable_op_name) != variables_.end()) {
    return variables_[variable_op_name];
  }
  return std::shared_ptr<one::Tensor>(nullptr);
}

Maybe<void> VariableTensorMgr::Delete(const std::string& variable_op_name) {
  if (variables_.find(variable_op_name) != variables_.end()) { variables_.erase(variable_op_name); }
  return Maybe<void>::Ok();
}

Maybe<void> VariableTensorMgr::Fill(
    const std::vector<std::string>& variable_op_names,
    const std::vector<std::shared_ptr<one::Tensor>>& variable_tensors) {
  CHECK_EQ_OR_THROW(variable_op_names.size(), variable_tensors.size())
      << "The number of variable op names is not equal with the number of variable tensors.";
  for (size_t i = 0; i < variable_op_names.size(); ++i) {
    JUST(Set(variable_op_names.at(i), variable_tensors.at(i)));
  }
  return Maybe<void>::Ok();
}

Maybe<std::tuple<std::vector<std::string>, std::vector<std::shared_ptr<one::Tensor>>>>
VariableTensorMgr::Dump() {
  std::vector<std::string> variable_op_names;
  std::vector<std::shared_ptr<one::Tensor>> variable_tensors;
  for (const auto& x : variables_) {
    variable_op_names.push_back(x.first);
    variable_tensors.push_back(x.second);
  }
  return std::make_tuple(variable_op_names, variable_tensors);
}

}  // namespace oneflow