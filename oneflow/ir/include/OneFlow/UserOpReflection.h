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
#ifndef ONEFLOW_IR_INCLUDE_ONEFLOW_USEROPRELFECTION_H_
#define ONEFLOW_IR_INCLUDE_ONEFLOW_USEROPRELFECTION_H_
#include "OneFlow/OneFlowOps.h"

namespace mlir {

namespace oneflow {

namespace user_op {

template<template<typename T> class Trait>
LogicalResult GetFilteredSegmentKeyAndSizes(Operation* op, std::vector<std::string>& keys,
                                            std::vector<int32_t>& sizes);
template<template<typename T> class Trait>
LogicalResult GetFilteredSegmentKeyAndSizes(llvm::StringRef op_type_name, size_t valueSize,
                                            DictionaryAttr attributes,
                                            std::vector<std::string>& keys,
                                            std::vector<int32_t>& sizes);

struct Source {
  enum {
    INPUT,
    OUTPUT,
    BUFFER,
    INVALID,
  } type;
  int offset;
};
Source GetOpSourceByName(Operation* op, const std::string& to_find);

using ArgID = std::pair<std::string, int32_t>;

template<template<typename T> class Trait>
class ArgIds {
 public:
  explicit ArgIds(Operation* op);
  ArgIds(llvm::StringRef op_type_name, size_t valueSize, DictionaryAttr attributes);
  std::vector<ArgID>::const_iterator begin() const { return ids_.begin(); }
  std::vector<ArgID>::const_iterator end() const { return ids_.end(); }

 private:
  std::vector<ArgID> ids_;
};

llvm::Optional<std::string> GetOutputLbn(OpResult result);

}  // namespace user_op

}  // namespace oneflow

}  // namespace mlir
#endif  // ONEFLOW_IR_INCLUDE_ONEFLOW_USEROPRELFECTION_H_
