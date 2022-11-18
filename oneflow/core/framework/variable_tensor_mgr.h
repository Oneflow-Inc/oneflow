/*
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
#ifndef ONEFLOW_CORE_FRAMEWORK_VARIABLE_TENSOR_MGR_H_
#define ONEFLOW_CORE_FRAMEWORK_VARIABLE_TENSOR_MGR_H_

#include <map>
#include <memory>
#include <tuple>
#include "oneflow/core/common/just.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/framework/dtype.h"
namespace oneflow {

template<typename T, typename Kind>
class Singleton;
namespace one {

class Tensor;

}

class VariableTensorMgr final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(VariableTensorMgr);
  ~VariableTensorMgr() = default;

  Maybe<void> Set(const std::string& variable_op_name,
                  const std::shared_ptr<one::Tensor>& variable_tensor,
                  const Symbol<DType>& dtype = Symbol<DType>());
  Maybe<one::Tensor> Get(const std::string& variable_op_name,
                         const Symbol<DType>& dtype = Symbol<DType>());

  void Delete(const std::string& variable_op_name);
  Maybe<void> Fill(const std::vector<std::string>& variable_op_names,
                   const std::vector<std::shared_ptr<one::Tensor>>& variable_tensors);
  std::tuple<std::vector<std::string>, std::vector<std::shared_ptr<one::Tensor>>> Dump();
  std::vector<std::string> DumpNames();
  void Reset();

 private:
  friend class Singleton<VariableTensorMgr>;
  VariableTensorMgr() = default;

  std::map<std::string, std::shared_ptr<one::Tensor>> variables_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_FRAMEWORK_VARIABLE_TENSOR_MGR_H_
