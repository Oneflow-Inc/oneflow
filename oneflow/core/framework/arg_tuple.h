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
#ifndef ONEFLOW_CORE_FRAMEWORK_ARG_TUPLE_H_
#define ONEFLOW_CORE_FRAMEWORK_ARG_TUPLE_H_

#include <string>
#include <vector>
#include <unordered_map>

namespace oneflow {

class ArgTuple final {
 public:
  explicit ArgTuple(const std::vector<std::string>& indexed_bns);
  ~ArgTuple() = default;

  std::size_t size() const { return indexed_bns_.size(); }

  const std::vector<std::string>& indexed_bns() const { return indexed_bns_; }
  const std::vector<std::pair<std::string, int32_t>>& indexed_arg_name_and_index() const {
    return indexed_arg_name_and_index_;
  }
  const std::unordered_map<std::string, std::vector<int32_t>>&
  arg_name2bn_index2tensor_tuple_index() const {
    return arg_name2bn_index2tensor_tuple_index_;
  }
  const std::unordered_map<std::string, int32_t>& bn_in_op2tensor_tuple_index() const {
    return bn_in_op2tensor_tuple_index_;
  }

  // return -1 if not found
  int32_t TensorTupleIndex4ArgNameAndIndex(const std::string& name, int32_t index) const;

 private:
  std::vector<std::string> indexed_bns_;
  std::vector<std::pair<std::string, int32_t>> indexed_arg_name_and_index_;
  std::unordered_map<std::string, std::vector<int32_t>> arg_name2bn_index2tensor_tuple_index_;
  std::unordered_map<std::string, int32_t> bn_in_op2tensor_tuple_index_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_FRAMEWORK_ARG_TUPLE_H_
