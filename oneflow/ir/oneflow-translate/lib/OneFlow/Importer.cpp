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
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/UseDefLists.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Translation.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm-c/Core.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/None.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"

#include "OneFlow/OneFlowDialect.h"
#include "OneFlow/OneFlowOps.h"
#include "OneFlow/MLIROneFlowTranslation.h"
#include "OneFlow/Passes.h"
#include "OneFlow/OneFlowSupport.h"

#include "oneflow/core/common/data_type.pb.h"
#include "oneflow/core/framework/user_op_conf.pb.h"
#include "oneflow/core/job/job.pb.h"
#include "oneflow/core/operator/op_conf.pb.h"
#include "oneflow/core/framework/user_op_def.h"
#include "oneflow/core/framework/user_op_registry_manager.h"
#include <cstddef>
#include <cstdint>
#include <google/protobuf/text_format.h>
#include <iostream>
#include <iterator>
#include <map>
#include <new>
#include <string>
#include <unordered_map>
#include <vector>

namespace mlir {

using PbMessage = google::protobuf::Message;

namespace {

using namespace ::oneflow;
const UserOpDef& GetUserOpDef(const std::string& op_type_name) {
  const user_op::OpRegistryResult* val =
      user_op::UserOpRegistryMgr::Get().GetOpRegistryResult(op_type_name);
  CHECK(val) << " Cannot find op_type_name: " << op_type_name;
  return val->op_def;
}

::oneflow::AttrType QueryAttrType(const std::string& op_type_name, const std::string& attr_name) {
  user_op::UserOpDefWrapper op_def(GetUserOpDef(op_type_name));
  CHECK(op_def.IsAttrName(attr_name)) << attr_name << " not a attr name for op: " << op_type_name;
  return op_def.GetAttrType(attr_name);
}

using SizeVec = SmallVector<int32_t, 8>;

SizeVec GetSizesFromArgs(UserOpArgs args, UserOpArgDefs arg_defs) {
  SizeVec sizes{};
  llvm::StringSet<> names({});
  for (const auto& arg : args) { names.insert(arg.first); }
  for (const auto& arg_def : arg_defs) {
    int32_t size = 0;
    if (names.contains(arg_def.name())) { size = args.at(arg_def.name()).s_size(); }
    sizes.push_back(size);
  }
  return sizes;
}

std::vector<std::string> GetOutputLbns(const ::oneflow::OperatorConf& op, UserOpArgDefs arg_defs) {
  SizeVec sizes{};
  llvm::StringSet<> names_appeared({});
  std::vector<std::string> output_lbn_vec{};
  const auto& op_name = op.name();
  for (const auto& arg : op.user_conf().output()) { names_appeared.insert(arg.first); }
  for (const auto& arg_def : arg_defs) {
    const auto& key = arg_def.name();
    auto result_size = op.user_conf().output().at(key).s_size();
    if (result_size == 0) { continue; }
    for (int32_t i = 0; i < result_size; i++) {
      const auto output_lbn = op_name + "/" + key + "_" + std::to_string(i);
      output_lbn_vec.push_back(output_lbn);
    }
  }
  return output_lbn_vec;
}

}  // namespace

LogicalResult Importer::AddUserOpInputOutputSegments(const ::oneflow::OperatorConf& op,
                                                     std::vector<NamedAttribute>& attr_vec) {
  if (op.has_user_conf() == false) return failure();
  const auto& user_conf = op.user_conf();
  const ::oneflow::UserOpDef& op_def = GetUserOpDef(op.user_conf().op_type_name());
  const auto UserOpOperationName =
      OperationName(oneflow::UserOp::getOperationName(), GetMLIRContext());
  attr_vec.push_back(GetBuilder().getNamedAttr(
      oneflow::UserOp::input_sizesAttrName(UserOpOperationName),
      GetBuilder().getI32ArrayAttr(GetSizesFromArgs(user_conf.input(), op_def.input()))));
  attr_vec.push_back(GetBuilder().getNamedAttr(
      oneflow::UserOp::output_sizesAttrName(UserOpOperationName),
      GetBuilder().getI32ArrayAttr(GetSizesFromArgs(user_conf.output(), op_def.output()))));
  auto output_lbns = GetOutputLbns(op, op_def.output());
  attr_vec.push_back(GetBuilder().getNamedAttr(
      OpTrait::IsImportCompatible<void>::getOutputLBNsAttr(),
      GetBuilder().getStrArrayAttr(
          SmallVector<StringRef, 8>({output_lbns.begin(), output_lbns.end()}))));
  return success();
}

OperandRange GetDataInputOperands(Operation* op) {
  if (auto cec = dyn_cast<ControlEdgeCompatible>(op)) {
    return cec.dataInputOperands();
  } else {
    return op->getOperands();
  }
}

llvm::Optional<OperandRange> GetCtrlIntputOperands(Operation* op) {
  if (auto cec = dyn_cast<ControlEdgeCompatible>(op)) {
    return cec.ctrlInputOperands();
  } else {
    return llvm::None;
  }
}

ResultRange GetDataOutputResults(Operation* op) {
  if (auto cec = dyn_cast<ControlEdgeCompatible>(op)) {
    return cec.dataOutputResults();
  } else {
    return op->getResults();
  }
}

llvm::Optional<OpResult> GetCtrlOutputResult(Operation* op) {
  if (auto cec = dyn_cast<ControlEdgeCompatible>(op)) {
    if (auto ctrl_out = cec.ctrlOutputResult()) { return ctrl_out.cast<OpResult>(); }
  }
  return llvm::None;
}

LogicalResult StringifyDataType(::oneflow::DataType value, std::string& stringified) {
  switch (value) {
    case ::oneflow::DataType::kInvalidDataType:
      stringified = stringifyEnum(oneflow::DataType::DT_InvalidDataType).str();
      break;
#define DEFINE_ONE_ELIF(datatype)                                        \
  case ::oneflow::DataType::k##datatype:                                 \
    stringified = stringifyEnum(oneflow::DataType::DT_##datatype).str(); \
    break;
      DEFINE_ONE_ELIF(Char)
      DEFINE_ONE_ELIF(Float)
      DEFINE_ONE_ELIF(Double)
      DEFINE_ONE_ELIF(Int8)
      DEFINE_ONE_ELIF(Int32)
      DEFINE_ONE_ELIF(Int64)
      DEFINE_ONE_ELIF(UInt8)
      DEFINE_ONE_ELIF(OFRecord)
      DEFINE_ONE_ELIF(Float16)
      DEFINE_ONE_ELIF(TensorBuffer)
#undef DEFINE_ONE_ELIF
    default: return failure();
  }
  return success();
}

DenseIntElementsAttr Importer::DenseIntElementsAttrFromShape(const ::oneflow::ShapeProto& shape) {
  ArrayRef<int64_t> values = {shape.dim().begin(), shape.dim().end()};
  RankedTensorType tt = RankedTensorType::get({static_cast<int64_t>(values.size())},
                                              GetBuilder().getIntegerType(64, true));
  ;
  return DenseIntElementsAttr::get(tt, values);
}

void WriteDenseIntElementsToShape(mlir::Attribute& attr, ::oneflow::ShapeProto* shape) {
  for (auto int_v : attr.dyn_cast<DenseIntElementsAttr>().getValues<int64_t>()) {
    shape->add_dim(int_v);
  }
}

LogicalResult Importer::namedAttributesFromUserOp(const ::oneflow::OperatorConf& op,
                                                  std::vector<NamedAttribute>& attr_vec) {
  if (op.has_user_conf() == false) {
    GetModule().emitError("Not a user op. op name: " + op.name());
    return failure();
  }
  for (const google::protobuf::MapPair<class std::basic_string<char>, ::oneflow::AttrValue>& attr :
       op.user_conf().attr()) {
    const std::string& name = attr.first;
    const ::oneflow::AttrValue& value = attr.second;
    if (value.has_at_int32()) {
      std::pair<mlir::Identifier, mlir::Attribute> kv =
          GetBuilder().getNamedAttr(name, GetBuilder().getSI32IntegerAttr(value.at_int32()));
      attr_vec.emplace_back(kv);
    } else if (value.has_at_int64()) {
      std::pair<mlir::Identifier, mlir::Attribute> kv =
          GetBuilder().getNamedAttr(name, getSI64IntegerAttr(value.at_int64()));
      attr_vec.emplace_back(kv);
    }
#define DEFINE_ONE_ELIF(at_key, get_attr)                                       \
  else if (value.has_##at_key()) {                                              \
    std::pair<mlir::Identifier, mlir::Attribute> kv =                           \
        GetBuilder().getNamedAttr(name, GetBuilder().get_attr(value.at_key())); \
    attr_vec.emplace_back(kv);                                                  \
  }
    DEFINE_ONE_ELIF(at_bool, getBoolAttr)
    DEFINE_ONE_ELIF(at_float, getF32FloatAttr)
    DEFINE_ONE_ELIF(at_double, getF64FloatAttr)
    DEFINE_ONE_ELIF(at_string, getStringAttr)
#undef DEFINE_ONE_ELIF
    else if (value.has_at_shape()) {
      attr_vec.emplace_back(
          GetBuilder().getNamedAttr(name, DenseIntElementsAttrFromShape(value.at_shape())));
    }
#define DEFINE_ONE_ELIF(at_key, get_attr, field)                                         \
  else if (value.has_##at_key()) {                                                       \
    std::pair<mlir::Identifier, mlir::Attribute> kv = GetBuilder().getNamedAttr(         \
        name, get_attr({value.at_key().field().begin(), value.at_key().field().end()})); \
    attr_vec.emplace_back(kv);                                                           \
  }
    DEFINE_ONE_ELIF(at_list_int32, getSI32ArrayAttr, val)
    DEFINE_ONE_ELIF(at_list_int64, getSI64ArrayAttr, val)
    DEFINE_ONE_ELIF(at_list_float, GetBuilder().getF32ArrayAttr, val)
#undef DEFINE_ONE_ELIF
    else if (value.has_at_list_string()) {
      std::vector<llvm::StringRef> r_vec = {value.at_list_string().val().begin(),
                                            value.at_list_string().val().end()};
      std::pair<mlir::Identifier, mlir::Attribute> kv =
          GetBuilder().getNamedAttr(name, GetBuilder().getStrArrayAttr(r_vec));
      attr_vec.emplace_back(kv);
    }
    else if (value.has_at_data_type()) {
      std::string stringified = "";
      if (failed(StringifyDataType(value.at_data_type(), stringified))) {
        GetModule().emitError("fail to convert op attr, key: " + name);
        return failure();
      }
      std::pair<mlir::Identifier, mlir::Attribute> kv =
          GetBuilder().getNamedAttr(name, GetBuilder().getStringAttr(stringified));
      attr_vec.emplace_back(kv);
    }
    else if (value.has_at_list_data_type()) {
      auto stringified_list = llvm::map_range(value.at_list_data_type().val(), [&](int32_t t) {
        std::string stringified = "";
        assert(succeeded(StringifyDataType(static_cast<::oneflow::DataType>(t), stringified)));
        return stringified;
      });
      std::vector<std::string> stringified_vector = {stringified_list.begin(),
                                                     stringified_list.end()};
      attr_vec.emplace_back(GetBuilder().getNamedAttr(
          name, GetBuilder().getStrArrayAttr(std::vector<StringRef>(
                    {stringified_vector.begin(), stringified_vector.end()}))));
    }
    else if (value.has_at_list_shape()) {
      auto dense_attr_list = llvm::map_range(
          value.at_list_shape().val(),
          [&](const ::oneflow::ShapeProto& s) { return DenseIntElementsAttrFromShape(s); });
      std::vector<mlir::Attribute> dense_attr_vector{dense_attr_list.begin(),
                                                     dense_attr_list.end()};
      attr_vec.emplace_back(
          GetBuilder().getNamedAttr(name, GetBuilder().getArrayAttr(dense_attr_vector)));
    }
    else {
      GetModule().emitError("can't handle user op attr: " + name + ", op name: " + op.name()
                            + ", op type name: " + op.user_conf().op_type_name());
      return failure();
    }
  }

  if (failed(AddUserOpInputOutputSegments(op, attr_vec))) {
    GetModule().emitError("fail to add input output segments: " + op.name());
    return failure();
  }

  return success();
}

LogicalResult Importer::AddOperandSegmentSizes(int32_t input_lbns_size, int32_t ctrl_in_size,
                                               std::vector<NamedAttribute>& attr_vec) {
  attr_vec.push_back(GetBuilder().getNamedAttr(
      mlir::OpTrait::AttrSizedOperandSegments<void>::getOperandSegmentSizeAttr(),
      GetBuilder().getI32VectorAttr({input_lbns_size, ctrl_in_size})));
  return success();
}

LogicalResult Importer::AddResultSegmentSizes(int32_t output_lbns_size,
                                              std::vector<NamedAttribute>& attr_vec) {
  attr_vec.push_back(GetBuilder().getNamedAttr(
      mlir::OpTrait::AttrSizedResultSegments<void>::getResultSegmentSizeAttr(),
      GetBuilder().getI32VectorAttr({output_lbns_size, 1} /* {data_out_size, ctrl_out_size} */)));
  return success();
}

LogicalResult Importer::AppendCtrlOutType(llvm::SmallVector<Type, 8>& out_types) {
  out_types.append({RankedTensorType::get({}, GetBuilder().getI1Type())});
  return success();
}

LogicalResult Importer::AddOpConf(const ::oneflow::OperatorConf& op,
                                  std::vector<NamedAttribute>& attr_vec) {
  attr_vec.push_back(GetBuilder().getNamedAttr(OpTrait::IsOpConfCompatible<void>::getOpNameAttr(),
                                               GetBuilder().getStringAttr(op.name())));
  if (op.has_device_tag()) {
    attr_vec.push_back(
        GetBuilder().getNamedAttr(OpTrait::IsOpConfCompatible<void>::getDeviceTagAttr(),
                                  GetBuilder().getStringAttr(op.device_tag())));
  }
  attr_vec.push_back(
      GetBuilder().getNamedAttr(OpTrait::IsOpConfCompatible<void>::getScopeSymbolIDAttr(),
                                GetBuilder().getI64IntegerAttr(op.scope_symbol_id())));
  return success();
}

llvm::Optional<Type> Importer::GetTypeFromOneFlowDataType(::oneflow::DataType dt) {
  {
    if (dt == ::oneflow::DataType::kInvalidDataType) { return llvm::None; }
    if (dt == ::oneflow::DataType::kChar) { return llvm::None; }
    if (dt == ::oneflow::DataType::kFloat) { return GetBuilder().getF32Type(); }
    if (dt == ::oneflow::DataType::kDouble) { return GetBuilder().getF64Type(); }
    if (dt == ::oneflow::DataType::kInt8) { return GetBuilder().getIntegerType(8, true); }
    if (dt == ::oneflow::DataType::kInt32) { return GetBuilder().getI32Type(); }
    if (dt == ::oneflow::DataType::kInt64) { return GetBuilder().getI64Type(); }
    if (dt == ::oneflow::DataType::kUInt8) { return GetBuilder().getIntegerType(8, false); }
    if (dt == ::oneflow::DataType::kOFRecord) { return llvm::None; }
    if (dt == ::oneflow::DataType::kFloat16) { return GetBuilder().getF16Type(); }
    if (dt == ::oneflow::DataType::kTensorBuffer) { return llvm::None; }
    return llvm::None;
  }
}

LogicalResult Importer::ProcessUserOp(const ::oneflow::OperatorConf& op) {
  if (op.has_user_conf() == false) {
    GetModule().emitError("Not a user op. op name: " + op.name());
    return failure();
  }
  std::vector<NamedAttribute> attr_vec;
  if (failed(AddOpConf(op, attr_vec))) { return failure(); }
  if (failed(AddDeviceName(op, attr_vec))) { return failure(); }
  attr_vec.push_back(
      GetBuilder().getNamedAttr(OpTrait::IsAlternative<void>::getOpTypeNameAttr(),
                                GetBuilder().getStringAttr(op.user_conf().op_type_name())));
  std::vector<::mlir::Value> operand_vec;
  if (failed(namedAttributesFromUserOp(op, attr_vec))) { return failure(); }
  const auto& op_def = GetUserOpDef(op.user_conf().op_type_name());
  for (const auto& arg_def : op_def.input()) {
    const auto& key = arg_def.name();
    auto it = op.user_conf().input().find(key);
    if (it == op.user_conf().input().end()) { continue; }
    int32_t index = 0;
    for (const std::string& lbn : it->second.s()) {
      if (failed(AppendDataInOperand(key, index, lbn, operand_vec))) { return failure(); }
      index += 1;
    }
  }
  if (failed(AppendCtrlInOperand(op, operand_vec))) { return failure(); }
  ::mlir::ValueRange operands(operand_vec);

  Operation* created_op = nullptr;

  auto out_types = llvm::SmallVector<Type, 8>();
  for (const auto& arg_def : op_def.output()) {
    const auto& key = arg_def.name();
    auto it = op.user_conf().output().find(key);
    if (it == op.user_conf().output().end()) { continue; }
    for (const auto& output_lbn : it->second.s()) {
      out_types.push_back(GetTensorTypeOfLbn(output_lbn));
    }
  }

  if (failed(AppendCtrlOutType(out_types))) { return failure(); }
  OperationState state(FileLineColLoc::get(GetMLIRContext(), op.name(), 0, 0), "oneflow.user");
  uint32_t data_input_size = 0;
  uint32_t data_output_size = 0;
  for (const auto& input : op.user_conf().input()) { data_input_size += input.second.s().size(); }
  for (const auto& output : op.user_conf().output()) {
    data_output_size += output.second.s().size();
  }
  if (failed(AddOperandSegmentSizes(data_input_size, op.ctrl_in_op_name_size(), attr_vec))) {
    return failure();
  }
  if (failed(AddResultSegmentSizes(data_output_size, attr_vec))) { return failure(); }
  ArrayRef<NamedAttribute> named_attributes(attr_vec);
  state.addAttributes(named_attributes);
  state.addOperands(operands);
  state.addTypes(out_types);
  created_op = GetBuilder().createOperation(state);

  if (created_op == nullptr) {
    GetModule()->emitError("fail to create " + op.user_conf().op_type_name()
                           + " op, name: " + op.name());
    return failure();
  }
  if (failed(InsertOpResults(op, created_op))) { return failure(); }

  return success();
}  // namespace

LogicalResult ConvertCtrlInputs(Operation* op, ::oneflow::OperatorConf& op_conf) {
  if (op->isRegistered() && !llvm::dyn_cast<oneflow::UserOp>(op)) return success();
  if (auto ctrl_ins = GetCtrlIntputOperands(op)) {
    for (auto ctrl_in : ctrl_ins.getValue()) {
      op_conf.add_ctrl_in_op_name(
          ctrl_in.getDefiningOp()
              ->getAttrOfType<StringAttr>(OpTrait::IsOpConfCompatible<void>::getOpNameAttr())
              .getValue()
              .str());
    }
  }
  return success();
}

template<template<typename T> class Trait>
const std::vector<std::string>* GetFullKeys(UserOpCompatible& uc, Operation* op);

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
LogicalResult GetFilteredSegmentKeyAndSizes(Operation* op, std::vector<std::string>& keys,
                                            std::vector<int32_t>& sizes) {
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
    assert(GetFilteredSegmentKeyAndSizes<OpTrait::AttrSizedResultSegments>(def_op, def_op_keys,
                                                                           def_op_sizes)
               .succeeded());
    const auto result_number = result.getResultNumber();
    uint32_t size_sum = 0;
    for (const auto& name_size_tuple : llvm::zip(def_op_keys, def_op_sizes)) {
      auto name = std::get<0>(name_size_tuple);
      auto size = std::get<1>(name_size_tuple);
      if ((size_sum + size) > result_number) {
        const uint32_t bn_i = result_number - size_sum;
        return def_op->getAttrOfType<StringAttr>(OpTrait::IsOpConfCompatible<void>::getOpNameAttr())
                   .str()
               + "/" + name + "_" + std::to_string(bn_i);
      }
      size_sum += size;
    }
  }
  return llvm::None;
}

LogicalResult ConvertUserOpInputs(Operation* op, oneflow::UserOpAdaptor& user_op_adaptor,
                                  ::oneflow::UserOpConf* user_conf) {
  std::vector<std::string> keys{};
  std::vector<int32_t> sizes{};
  assert(GetFilteredSegmentKeyAndSizes<OpTrait::AttrSizedOperandSegments>(op, keys, sizes)
             .succeeded());
  const std::string op_name = user_op_adaptor.op_name().getValue().str();
  int32_t input_idx = 0;
  for (auto tuple : llvm::zip(keys, sizes)) {
    auto input_key = std::get<0>(tuple);
    auto input_size = std::get<1>(tuple);
    assert(input_size > 0);
    for (int32_t i = 0; i < input_size; i++) {
      if (auto result = op->getOperand(input_idx).dyn_cast<mlir::OpResult>()) {
        auto input_s_ptr = (*user_conf->mutable_input())[input_key].mutable_s()->Add();
        *(input_s_ptr) = GetOutputLbn(result).getValue();
        input_idx += 1;
      } else {
        op->emitError() << "fail to convert MLIR result to protobuf, name: " + op_name;
        op->dump();
        return failure();
      }
    }
  }
  return success();
}

LogicalResult ConvertUserOpOutputs(Operation* op, oneflow::UserOpAdaptor& user_op_adaptor,
                                   ::oneflow::UserOpConf* user_conf) {
  std::vector<std::string> keys{};
  std::vector<int32_t> sizes{};
  assert(
      GetFilteredSegmentKeyAndSizes<OpTrait::AttrSizedResultSegments>(op, keys, sizes).succeeded());
  const std::string op_name = user_op_adaptor.op_name().getValue().str();
  int32_t result_idx = 0;
  for (auto tuple : llvm::zip(keys, sizes)) {
    auto name = std::get<0>(tuple);
    auto result_size = std::get<1>(tuple);
    if (result_size == 0) continue;
    for (int32_t i = 0; i < result_size; i++) {
      auto out_s_ptr = (*user_conf->mutable_output())[name].mutable_s()->Add();
      *(out_s_ptr) = op_name + "/" + name + "_" + std::to_string(i);
      result_idx += 1;
    }
  }
  return success();
}

LogicalResult ConvertDT(Attribute attr, ::oneflow::DataType& data_type) {
  Optional<mlir::oneflow::DataType> dt =
      oneflow::symbolizeEnum<oneflow::DataType>(attr.dyn_cast<StringAttr>().getValue().trim());
  assert(dt.hasValue());
  switch (dt.getValue()) {
    case oneflow::DataType::DT_InvalidDataType:
      data_type = ::oneflow::DataType::kInvalidDataType;
      break;
#define DEFINE_ONE_CASE(datatype) \
  case oneflow::DataType::DT_##datatype: data_type = ::oneflow::DataType::k##datatype; break;
      DEFINE_ONE_CASE(Char)
      DEFINE_ONE_CASE(Float)
      DEFINE_ONE_CASE(Double)
      DEFINE_ONE_CASE(Int8)
      DEFINE_ONE_CASE(Int32)
      DEFINE_ONE_CASE(Int64)
      DEFINE_ONE_CASE(UInt8)
      DEFINE_ONE_CASE(OFRecord)
      DEFINE_ONE_CASE(Float16)
      DEFINE_ONE_CASE(TensorBuffer)
#undef DEFINE_ONE_CASE
    default: return failure();
  }
  return success();
}

LogicalResult Importer::ConvertUserOpAttributes(Operation* op,
                                                oneflow::UserOpAdaptor& user_op_adaptor,
                                                ::oneflow::OperatorConf& op_conf) {
  auto user_conf = op_conf.mutable_user_conf();
  std::string op_type_name = GetOpTypeName(op);
  op_conf.mutable_user_conf()->set_op_type_name(op_type_name);
  for (auto id_attr : op->getAttrDictionary()) {
    auto id = id_attr.first;
    // mlir only attrs
    // TODO: find a way to skip attrs like callee in a declarative way
    {
      std::vector<std::string> keys{};
      std::vector<int32_t> sizes{};
      assert(GetFilteredSegmentKeyAndSizes<OpTrait::AttrSizedOperandSegments>(op, keys, sizes)
                 .succeeded());
      for (const auto& s : keys) { op_conf.mutable_user_conf()->add_input_order(s); }
    }
    {
      std::vector<std::string> keys{};
      std::vector<int32_t> sizes{};
      assert(GetFilteredSegmentKeyAndSizes<OpTrait::AttrSizedResultSegments>(op, keys, sizes)
                 .succeeded());
      for (const auto& s : keys) { op_conf.mutable_user_conf()->add_output_order(s); }
    }
    if (id.strref().equals("callee")
        || id.strref().equals(OpTrait::IsOpConfCompatible<void>::getDeviceNameAttr())
        || id.strref().equals(OpTrait::IsOpConfCompatible<void>::getHierarchyAttr())
        || id.strref().equals(OpTrait::IsImportCompatible<void>::getOutputLBNsAttr())
        || id.strref().equals(OpTrait::IsAlternative<void>::getOpTypeNameAttr())
        || id.strref().equals(
            mlir::OpTrait::AttrSizedOperandSegments<void>::getOperandSegmentSizeAttr())
        || id.strref().equals(
            mlir::OpTrait::AttrSizedResultSegments<void>::getResultSegmentSizeAttr())) {
      continue;
    }
    // convert op conf attributes
    else if (id.strref().equals(OpTrait::IsOpConfCompatible<void>::getOpNameAttr())) {
      std::string op_name =
          op->getAttrOfType<StringAttr>(OpTrait::IsOpConfCompatible<void>::getOpNameAttr())
              .getValue()
              .str();
      op_conf.set_name(op_name);
    } else if (id.strref().equals(OpTrait::IsOpConfCompatible<void>::getDeviceTagAttr())) {
      op_conf.set_device_tag(user_op_adaptor.device_tag().getValue().str());
    } else if (id.strref().equals(OpTrait::IsOpConfCompatible<void>::getScopeSymbolIDAttr())) {
      op_conf.set_scope_symbol_id(user_op_adaptor.scope_symbol_id().getInt());
    }
    // convert user conf attributes
    else {
      auto attr_name = id.str();
      Attribute attr = id_attr.second;
      auto user_attr = ::oneflow::AttrValue();
      const ::oneflow::AttrType attr_type = QueryAttrType(op_type_name, attr_name);
      if (attr_type == ::oneflow::kAtInt32) {
        user_attr.set_at_int32(attr.dyn_cast<IntegerAttr>().getSInt());
      } else if (attr_type == ::oneflow::kAtInt64) {
        user_attr.set_at_int64(attr.dyn_cast<IntegerAttr>().getSInt());
      } else if (attr_type == ::oneflow::kAtBool) {
        user_attr.set_at_bool(attr.dyn_cast<BoolAttr>().getValue());
      } else if (attr_type == ::oneflow::kAtFloat) {
        user_attr.set_at_float(attr.dyn_cast<FloatAttr>().getValue().convertToFloat());
      } else if (attr_type == ::oneflow::kAtDouble) {
        user_attr.set_at_double(attr.dyn_cast<FloatAttr>().getValue().convertToDouble());
      } else if (attr_type == ::oneflow::kAtString) {
        user_attr.set_at_string(attr.dyn_cast<StringAttr>().getValue().str());
      } else if (attr_type == ::oneflow::kAtShape) {
        WriteDenseIntElementsToShape(attr, user_attr.mutable_at_shape());
      } else if (attr_type == ::oneflow::kAtDataType) {
        ::oneflow::DataType dt = ::oneflow::kInvalidDataType;
        if (succeeded(ConvertDT(attr, dt))) {
          user_attr.set_at_data_type(dt);
        } else {
          op->emitError() << "fail to convert op attr to data type, key: " + id.str();
          return failure();
        }
      } else if (attr_type == ::oneflow::kAtListInt32) {
        user_attr.mutable_at_list_int32();
        auto ref = attr.dyn_cast<ArrayAttr>();
        for (auto v : ref.getValue()) {
          user_attr.mutable_at_list_int32()->add_val(v.dyn_cast<IntegerAttr>().getSInt());
        }
      } else if (attr_type == ::oneflow::kAtListInt64) {
        user_attr.mutable_at_list_int64();
        auto ref = attr.dyn_cast<ArrayAttr>();
        for (auto v : ref.getValue()) {
          user_attr.mutable_at_list_int64()->add_val(v.dyn_cast<IntegerAttr>().getSInt());
        }
      } else if (attr_type == ::oneflow::kAtListFloat) {
        user_attr.mutable_at_list_float();
        auto ref = attr.dyn_cast<ArrayAttr>();
        for (auto v : ref.getValue()) {
          user_attr.mutable_at_list_float()->add_val(
              v.dyn_cast<FloatAttr>().getValue().convertToFloat());
        }
      } else if (attr_type == ::oneflow::kAtListDataType) {
        for (auto v : attr.dyn_cast<ArrayAttr>().getValue()) {
          ::oneflow::DataType dt = ::oneflow::kInvalidDataType;
          if (succeeded(ConvertDT(v, dt))) {
            user_attr.mutable_at_list_data_type()->add_val(dt);
          } else {
            op->emitError() << "fail to convert op attr to data type, key: " + id.str();
            return failure();
          }
        }
      } else if (attr_type == ::oneflow::kAtListShape) {
        for (auto s : attr.dyn_cast<ArrayAttr>().getValue()) {
          ::oneflow::ShapeProto* shape_ptr = user_attr.mutable_at_list_shape()->add_val();
          for (auto int_v : s.dyn_cast<DenseIntElementsAttr>().getValues<int64_t>()) {
            shape_ptr->mutable_dim()->Add(int_v);
          }
        }
      } else if (attr_type == ::oneflow::kAtListString) {
        // attr like nd_sbp requires the existence of list even it is empty
        user_attr.mutable_at_list_string();
        for (auto s : attr.dyn_cast<ArrayAttr>().getValue()) {
          user_attr.mutable_at_list_string()->add_val(s.dyn_cast<StringAttr>().getValue().str());
        }
      } else {
        op->emitError() << "fail to convert op attr of name: " + attr_name;
        return failure();
      }
      (*user_conf->mutable_attr())[id.str()] = user_attr;
    }
  }
  return success();
}

}  // namespace mlir
