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
// this file should contains functions to get operands and results with user op name and index

#include "OneFlow/UserOpReflection.h"

namespace mlir {

namespace oneflow {

namespace user_op {

template<template<typename T> class Trait>
const std::vector<std::string>* GetFullKeys(UserOpCompatible& uc, Operation* op);
template<template<typename T> class Trait>
std::vector<std::string> GetFullKeys(UserOp op);

template<>
const std::vector<std::string>* GetFullKeys<OpTrait::AttrSizedOperandSegments>(UserOpCompatible& uc,
                                                                               Operation* op) {
  if (auto alternative_name = dyn_cast<HasAlternativeOpTypeName>(op)) {
    return alternative_name.inputKeys();
  }
  return uc.inputKeys();
}

template<>
const std::vector<std::string>* GetFullKeys<OpTrait::AttrSizedResultSegments>(UserOpCompatible& uc,
                                                                              Operation* op) {
  if (auto alternative_name = dyn_cast<HasAlternativeOpTypeName>(op)) {
    return alternative_name.outputKeys();
  }
  return uc.outputKeys();
}

template<>
std::vector<std::string> GetFullKeys<OpTrait::AttrSizedOperandSegments>(UserOp op) {
  return mlir::oneflow::support::GetInputKeys(op.op_type_name().str());
}

template<>
std::vector<std::string> GetFullKeys<OpTrait::AttrSizedResultSegments>(UserOp op) {
  return mlir::oneflow::support::GetOutputKeys(op.op_type_name().str());
}

template<template<typename T> class Trait>
std::pair<unsigned, unsigned> getODSIndexAndLength(UserOpCompatible& op, unsigned index);

template<>
std::pair<unsigned, unsigned> getODSIndexAndLength<OpTrait::AttrSizedOperandSegments>(
    UserOpCompatible& op, unsigned index) {
  return op.getODSOperandIndexAndLength(index);
}

template<>
std::pair<unsigned, unsigned> getODSIndexAndLength<OpTrait::AttrSizedResultSegments>(
    UserOpCompatible& op, unsigned index) {
  return op.getODSResultIndexAndLength(index);
}

template<template<typename T> class Trait>
StringRef GetSegmentSizeAttr();

template<>
StringRef GetSegmentSizeAttr<OpTrait::AttrSizedOperandSegments>() {
  return OpTrait::AttrSizedOperandSegments<void>::getOperandSegmentSizeAttr();
}

template<>
StringRef GetSegmentSizeAttr<OpTrait::AttrSizedResultSegments>() {
  return OpTrait::AttrSizedResultSegments<void>::getResultSegmentSizeAttr();
}

template<template<typename T> class Trait>
int32_t GetSingleSegmentSize(Operation*);

template<>
int32_t GetSingleSegmentSize<OpTrait::AttrSizedOperandSegments>(Operation* op) {
  return op->getNumOperands();
}

template<>
int32_t GetSingleSegmentSize<OpTrait::AttrSizedResultSegments>(Operation* op) {
  return op->getNumResults();
}

template<template<typename T> class Trait>
ArrayAttr GetUserOpArgSizes(UserOp);

template<>
ArrayAttr GetUserOpArgSizes<OpTrait::AttrSizedOperandSegments>(UserOp op) {
  return op.input_sizes();
}

template<>
ArrayAttr GetUserOpArgSizes<OpTrait::AttrSizedResultSegments>(UserOp op) {
  return op.output_sizes();
}

template<template<typename T> class Trait>
LogicalResult GetUserOpFilteredSegmentKeyAndSizes(UserOp op, std::vector<std::string>& keys,
                                                  std::vector<int32_t>& sizes) {
  auto full_keys = GetFullKeys<Trait>(op);
  for (const auto& key_size_tuple : llvm::zip(full_keys, GetUserOpArgSizes<Trait>(op).getValue())) {
    const std::string& key = std::get<0>(key_size_tuple);
    const int32_t size =
        std::get<1>(key_size_tuple).template cast<IntegerAttr>().getValue().getSExtValue();
    if (size > 0) {
      keys.push_back(key);
      sizes.push_back(size);
    }
  }
  return success();
}

template<template<typename T> class Trait>
LogicalResult GetFilteredSegmentKeyAndSizes(Operation* op, std::vector<std::string>& keys,
                                            std::vector<int32_t>& sizes) {
  if (auto user_op = dyn_cast<UserOp>(op)) {
    return GetUserOpFilteredSegmentKeyAndSizes<Trait>(user_op, keys, sizes);
  }
  const std::vector<std::string>* full_keys = nullptr;
  std::vector<int32_t> full_sizes{};
  auto uc = dyn_cast<UserOpCompatible>(op);
  if (!uc) {
    op->emitError("interface UserOpCompatible not supported");
    return failure();
  }
  full_keys = GetFullKeys<Trait>(uc, op);
  if (op->hasTrait<Trait>()) {
    const StringRef attr_name = GetSegmentSizeAttr<Trait>();
    const DenseIntElementsAttr& size_attr = op->getAttrOfType<DenseIntElementsAttr>(attr_name);
    if (!size_attr) return failure();
    auto segment_sizes = size_attr.getValues<int32_t>();
    if (full_keys->size() != segment_sizes.size()) {
      op->emitError() << "fail to convert op inputs, attr_name: " << attr_name
                      << ", full_keys: " << full_keys->size()
                      << ", segment_sizes: " << segment_sizes.size() << ", name: " << op->getName();
      op->dump();
      return failure();
    };
    full_sizes = {segment_sizes.begin(), segment_sizes.end()};
  } else {
    if (full_keys->size() == 1) {
      full_sizes.push_back(GetSingleSegmentSize<Trait>(op));
    } else {
      for (const auto& key : llvm::enumerate(*full_keys)) {
        full_sizes.push_back(getODSIndexAndLength<Trait>(uc, key.index()).second);
      }
    }
  }
  for (const auto& key_size_tuple : llvm::zip(*full_keys, full_sizes)) {
    const std::string& key = std::get<0>(key_size_tuple);
    const int32_t size = std::get<1>(key_size_tuple);
    if (size > 0) {
      keys.push_back(key);
      sizes.push_back(size);
    }
  }
  return success();
}

template LogicalResult GetFilteredSegmentKeyAndSizes<OpTrait::AttrSizedOperandSegments>(
    Operation* op, std::vector<std::string>& keys, std::vector<int32_t>& sizes);
template LogicalResult GetFilteredSegmentKeyAndSizes<OpTrait::AttrSizedResultSegments>(
    Operation* op, std::vector<std::string>& keys, std::vector<int32_t>& sizes);

template<template<typename T> class Trait>
ArgIds<Trait>::ArgIds(Operation* op) {
  std::vector<std::string> keys;
  std::vector<int32_t> sizes;
  if (failed(GetFilteredSegmentKeyAndSizes<Trait>(op, keys, sizes))) {
    op->emitError("fail to get filtered segment key and sizes");
    exit(1);
  }
  for (int i = 0; i < keys.size(); i += 1) {
    auto& key = keys[i];
    for (size_t j = 0; j < sizes.size(); j += 1) {
      ArgID id{key, j};
      ids_.push_back(id);
    }
  }
}

template oneflow::user_op::ArgIds<OpTrait::AttrSizedOperandSegments>::ArgIds(Operation*);
template oneflow::user_op::ArgIds<OpTrait::AttrSizedResultSegments>::ArgIds(Operation*);

}  // namespace user_op

}  // namespace oneflow

}  // namespace mlir
