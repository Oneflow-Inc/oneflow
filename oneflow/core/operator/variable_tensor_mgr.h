#ifndef ONEFLOW_CORE_OPERATOR_VARIABLE_TENSOR_MGR_H_
#define ONEFLOW_CORE_OPERATOR_VARIABLE_TENSOR_MGR_H_

#include <map>
#include <memory>
#include <tuple>
#include "oneflow/core/common/just.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/common/global.h"
#include "oneflow/core/framework/tensor.h"

namespace oneflow {

class VariableTensorMgr final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(VariableTensorMgr);
  ~VariableTensorMgr() = default;

  Maybe<void> Set(const std::string& variable_op_name,
                  const std::shared_ptr<one::Tensor>& variable_tensor);
  Maybe<one::Tensor> Get(const std::string& variable_op_name);
  Maybe<void> Delete(const std::string& variable_op_name);
  Maybe<void> Fill(const std::vector<std::string>& variable_op_names,
                   const std::vector<std::shared_ptr<one::Tensor>>& variable_tensors);
  Maybe<std::tuple<std::vector<std::string>, std::vector<std::shared_ptr<one::Tensor>>>> Dump();

 private:
  friend class Global<VariableTensorMgr>;
  VariableTensorMgr() = default;

  std::map<std::string, std::shared_ptr<one::Tensor>> variables_;
};

}  // namespace oneflow

#endif // ONEFLOW_CORE_OPERATOR_VARIABLE_TENSOR_MGR_H_
