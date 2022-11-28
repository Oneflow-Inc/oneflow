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

#include "oneflow/core/framework/arg_tuple.h"
#include <glog/logging.h>

namespace oneflow {

namespace {

std::pair<std::string, int> GetPair(const std::string& bn) {
  int32_t index = 0;
  const size_t pos = bn.rfind('_');
  if (pos != std::string::npos) { index = std::stoi(bn.substr(pos + 1)); }
  return std::make_pair(bn.substr(0, pos), index);
}

void InitArgName2BnIndex2TensorTupleIndex(
    const std::vector<std::pair<std::string, int32_t>>& indexed_arg_pairs,
    std::unordered_map<std::string, std::vector<int32_t>>* arg_name2bn_index2tensor_tuple_index) {
  for (int i = 0; i < indexed_arg_pairs.size(); i++) {
    const auto& pair = indexed_arg_pairs.at(i);
    const std::string& arg_name = pair.first;
    const int32_t bn_index = pair.second;
    // vector is auto created by [] if arg_name doesn't exist in map
    auto* bn_index2tensor_tuple_index = &(*arg_name2bn_index2tensor_tuple_index)[arg_name];
    CHECK_EQ(bn_index2tensor_tuple_index->size(), bn_index)
        << "Duplicate index of " << arg_name << ": " << bn_index;
    bn_index2tensor_tuple_index->emplace_back(i);
  }
}

}  // namespace

ArgTuple::ArgTuple(const std::vector<std::string>& indexed_bns) : indexed_bns_(indexed_bns) {
  indexed_arg_name_and_index_.reserve(indexed_bns.size());
  for (const auto& bn : indexed_bns) { indexed_arg_name_and_index_.emplace_back(GetPair(bn)); }
  InitArgName2BnIndex2TensorTupleIndex(indexed_arg_name_and_index_,
                                       &arg_name2bn_index2tensor_tuple_index_);
  for (int i = 0; i < indexed_bns.size(); ++i) {
    bn_in_op2tensor_tuple_index_[indexed_bns.at(i)] = i;
  }
}

int32_t ArgTuple::TensorTupleIndex4ArgNameAndIndex(const std::string& name, int32_t index) const {
  const auto& map = arg_name2bn_index2tensor_tuple_index_;
  const auto& iter = map.find(name);
  if (iter == map.end()) { return -1; }
  const auto& vec = iter->second;
  return vec.at(index);
}

}  // namespace oneflow
