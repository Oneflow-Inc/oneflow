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
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Casting.h"

namespace mlir {

namespace oneflow {

namespace user_op {

template<template<typename T> class Trait>
const std::vector<std::string>* GetFullKeys(UserOpCompatible& uc, Operation* op);
template<template<typename T> class Trait>
std::vector<std::string> GetFullKeys(UserOp op);
template<template<typename T> class Trait>
std::vector<std::string> GetFullKeys(::llvm::StringRef op_type_name);

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

template<>
std::vector<std::string> GetFullKeys<OpTrait::AttrSizedOperandSegments>(
    ::llvm::StringRef op_type_name) {
  return mlir::oneflow::support::GetInputKeys(op_type_name.str());
}

template<>
std::vector<std::string> GetFullKeys<OpTrait::AttrSizedResultSegments>(
    ::llvm::StringRef op_type_name) {
  return mlir::oneflow::support::GetOutputKeys(op_type_name.str());
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

Source GetOpSourceByName(Operation* op, const std::string& to_find) {
  if (auto user_op = dyn_cast<UserOpCompatible>(op)) {
    auto found = [&](std::vector<std::string> keys,
                     bool find_in_results /*or in operands*/ = false) -> int {
      auto offset = 0;
      for (const auto& key : llvm::enumerate(keys)) {
        if (key.value() == to_find) { return offset; }
        offset += find_in_results ? user_op.getODSResultIndexAndLength(key.index()).second
                                  : user_op.getODSOperandIndexAndLength(key.index()).second;
      }
      return -1;
    };

    if (auto alternative_name = dyn_cast<HasAlternativeOpTypeName>(op)) {
      if (auto offset = found(*alternative_name.inputKeys()); offset != -1) {
        return {Source::INPUT, offset};
      }
      if (auto offset = found(*alternative_name.outputKeys(), true); offset != -1) {
        return {Source::OUTPUT, offset};
      }
    }

    if (to_find == "tmp_buffer") { return {Source::BUFFER, 0}; }

    if (auto offset = found(*user_op.inputKeys()); offset != -1) { return {Source::INPUT, offset}; }
    if (auto offset = found(*user_op.outputKeys(), true); offset != -1) {
      return {Source::OUTPUT, offset};
    }

    op->emitError(to_find + " not found in this op");
    return {Source::INVALID, -1};
  }
  op->emitError("Not support op which is not user  op");
  return {Source::INVALID, -1};
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

template<template<typename T> class Trait>
LogicalResult GetFilteredSegmentKeyAndSizes(llvm::StringRef op_type_name, size_t valueSize,
                                            DictionaryAttr attributes,
                                            std::vector<std::string>& keys,
                                            std::vector<int32_t>& sizes) {
  const std::vector<std::string> full_keys = GetFullKeys<Trait>(op_type_name);
  std::vector<int32_t> full_sizes{};
  const StringRef attr_name = GetSegmentSizeAttr<Trait>();
  if (auto size_attr = attributes.get(attr_name).dyn_cast_or_null<DenseIntElementsAttr>()) {
    if (!size_attr) return failure();
    auto segment_sizes = size_attr.getValues<int32_t>();
    if (full_keys.size() != segment_sizes.size()) {
      LOG(FATAL) << "fail to convert op inputs, attr_name: " << attr_name.str()
                 << ", full_keys: " << full_keys.size()
                 << ", segment_sizes: " << segment_sizes.size();
      return failure();
    };
    full_sizes = {segment_sizes.begin(), segment_sizes.end()};
  } else {
    if (full_keys.size() == 1) {
      full_sizes.push_back(valueSize);
    } else {
      LOG(FATAL) << "set attr: " << attr_name.str();
    }
  }
  for (const auto& key_size_tuple : llvm::zip(full_keys, full_sizes)) {
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
template LogicalResult GetFilteredSegmentKeyAndSizes<OpTrait::AttrSizedOperandSegments>(
    llvm::StringRef op_type_name, size_t valueSize, DictionaryAttr attributes,
    std::vector<std::string>& keys, std::vector<int32_t>& sizes);
template LogicalResult GetFilteredSegmentKeyAndSizes<OpTrait::AttrSizedResultSegments>(
    llvm::StringRef op_type_name, size_t valueSize, DictionaryAttr attributes,
    std::vector<std::string>& keys, std::vector<int32_t>& sizes);

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
    for (size_t j = 0; j < sizes[i]; j += 1) {
      ArgID id{key, j};
      ids_.push_back(id);
    }
  }
}

template<template<typename T> class Trait>
ArgIds<Trait>::ArgIds(llvm::StringRef op_type_name, size_t valueSize, DictionaryAttr attributes) {
  std::vector<std::string> keys{};
  std::vector<int32_t> sizes{};
  CHECK(user_op::GetFilteredSegmentKeyAndSizes<Trait>(op_type_name, valueSize, attributes, keys,
                                                      sizes)
            .succeeded());
  for (int i = 0; i < keys.size(); i += 1) {
    auto& key = keys[i];
    for (size_t j = 0; j < sizes[i]; j += 1) {
      ArgID id{key, j};
      ids_.push_back(id);
    }
  }
}

template oneflow::user_op::ArgIds<OpTrait::AttrSizedOperandSegments>::ArgIds(Operation*);
template oneflow::user_op::ArgIds<OpTrait::AttrSizedResultSegments>::ArgIds(Operation*);
template oneflow::user_op::ArgIds<OpTrait::AttrSizedOperandSegments>::ArgIds(
    llvm::StringRef op_type_name, size_t valueSize, DictionaryAttr attributes);
template oneflow::user_op::ArgIds<OpTrait::AttrSizedResultSegments>::ArgIds(
    llvm::StringRef op_type_name, size_t valueSize, DictionaryAttr attributes);

llvm::Optional<std::string> GetOutputLbn(OpResult result) {
  const auto def_op = result.getDefiningOp();
  if (def_op->hasTrait<OpTrait::IsImportCompatible>()) {
    return def_op
        ->getAttrOfType<ArrayAttr>(
            OpTrait::IsImportCompatible<void>::getOutputLBNsAttr())[result.getResultNumber()]
        .dyn_cast<StringAttr>()
        .getValue()
        .str();
  } else {
    std::vector<std::string> def_op_keys{};
    std::vector<int32_t> def_op_sizes{};
    if (failed(user_op::GetFilteredSegmentKeyAndSizes<OpTrait::AttrSizedResultSegments>(
            def_op, def_op_keys, def_op_sizes))) {
      def_op->emitError("fail to get output lbn");
      return llvm::None;
    }
    const auto result_number = result.getResultNumber();
    uint32_t size_sum = 0;
    for (const auto& name_size_tuple : llvm::zip(def_op_keys, def_op_sizes)) {
      auto name = std::get<0>(name_size_tuple);
      auto size = std::get<1>(name_size_tuple);
      if ((size_sum + size) > result_number) {
        const uint32_t bn_i = result_number - size_sum;
        return OpTrait::IsOpConfCompatible<void>::getOpName(def_op).str() + "/" + name + "_"
               + std::to_string(bn_i);
      }
      size_sum += size;
    }
  }
  return llvm::None;
}

}  // namespace user_op

}  // namespace oneflow

}  // namespace mlir
