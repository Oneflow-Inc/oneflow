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
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"

#include "OneFlow/OneFlowDialect.h"
#include "OneFlow/OneFlowOps.h"
#include "OneFlow/MLIROneFlowTranslation.h"
#include "OneFlow/Passes.h"

#include "oneflow/core/common/data_type.pb.h"
#include "oneflow/core/framework/user_op_conf.pb.h"
#include "oneflow/core/job/job.pb.h"
#include "oneflow/core/operator/op_conf.pb.h"
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

LogicalResult Importer::AddUserOpInputOutputSegments(const ::oneflow::OperatorConf& op,
                                                     std::vector<NamedAttribute>& attr_vec) {
  using LBNVec = SmallVector<StringRef, 8>;
  using LBNSegVec = SmallVector<int32_t, 8>;
  LBNVec input_lbn_segment_keys;
  LBNSegVec input_lbn_segment_sizes;
  int32_t data_input_size = 0;
  for (const auto& key : op.user_conf().input_order()) {
    auto& value = op.user_conf().input().at(key);
    input_lbn_segment_keys.push_back(key);
    input_lbn_segment_sizes.push_back(value.s_size());
    data_input_size += value.s_size();
  }
  attr_vec.push_back(GetBuilder().getNamedAttr(
      "input_lbn_segment_keys", GetBuilder().getStrArrayAttr(input_lbn_segment_keys)));
  attr_vec.push_back(GetBuilder().getNamedAttr(
      "input_lbn_segment_sizes", GetBuilder().getI32ArrayAttr(input_lbn_segment_sizes)));

  LBNVec output_lbns;
  LBNVec output_lbn_segment_keys;
  LBNSegVec output_lbn_segment_sizes;
  int32_t data_output_size = 0;
  for (const auto& key : op.user_conf().output_order()) {
    auto& value = op.user_conf().output().at(key);
    output_lbns.insert(output_lbns.end(), value.s().begin(), value.s().end());
    output_lbn_segment_keys.push_back(key);
    output_lbn_segment_sizes.push_back(value.s_size());
    data_output_size += value.s_size();
  }
  attr_vec.push_back(
      GetBuilder().getNamedAttr("output_lbns", GetBuilder().getStrArrayAttr(output_lbns)));
  attr_vec.push_back(GetBuilder().getNamedAttr(
      "output_lbn_segment_keys", GetBuilder().getStrArrayAttr(output_lbn_segment_keys)));
  attr_vec.push_back(GetBuilder().getNamedAttr(
      "output_lbn_segment_sizes", GetBuilder().getI32ArrayAttr(output_lbn_segment_sizes)));
  return success();
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
  for (auto int_v : attr.dyn_cast<DenseIntElementsAttr>().getIntValues()) {
    assert(int_v.isSignedIntN(64));
    shape->add_dim(int_v.getSExtValue());
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
      "operand_segment_sizes", GetBuilder().getI32VectorAttr({input_lbns_size, ctrl_in_size})));
  return success();
}

LogicalResult Importer::AddResultSegmentSizes(int32_t output_lbns_size,
                                              std::vector<NamedAttribute>& attr_vec) {
  attr_vec.push_back(GetBuilder().getNamedAttr(
      "result_segment_sizes",
      GetBuilder().getI32VectorAttr({output_lbns_size, 1} /* {data_out_size, ctrl_out_size} */)));
  return success();
}

std::pair<unsigned, unsigned> getODSOperandIndexAndLength(Operation* op, unsigned index) {
  auto sizeAttr = op->getAttrOfType<::mlir::DenseIntElementsAttr>("operand_segment_sizes");

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i) start += (*(sizeAttr.begin() + i)).getZExtValue();
  unsigned size = (*(sizeAttr.begin() + index)).getZExtValue();
  return {start, size};
}

::mlir::Operation::operand_range getODSOperands(Operation* op, unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(op, index);
  return {std::next(op->operand_begin(), valueRange.first),
          std::next(op->operand_begin(), valueRange.first + valueRange.second)};
}

OperandRange GetDataInputOperands(Operation* op) {
  if (op->hasAttrOfType<::mlir::DenseIntElementsAttr>("operand_segment_sizes")) {
    return getODSOperands(op, 0);
  } else {
    return op->getOperands();
  }
}

llvm::Optional<OperandRange> GetCtrlIntputOperands(Operation* op) {
  if (op->hasAttrOfType<::mlir::DenseIntElementsAttr>("operand_segment_sizes")) {
    return getODSOperands(op, 1);
  } else {
    return llvm::None;
  }
}

LogicalResult Importer::AppendCtrlOutType(llvm::SmallVector<Type, 8>& out_types) {
  out_types.append({RankedTensorType::get({}, GetBuilder().getI1Type())});
  return success();
}

LogicalResult Importer::AddOpConf(const ::oneflow::OperatorConf& op,
                                  std::vector<NamedAttribute>& attr_vec) {
  attr_vec.push_back(GetBuilder().getNamedAttr("op_name", GetBuilder().getStringAttr(op.name())));
  if (op.has_device_tag()) {
    attr_vec.push_back(
        GetBuilder().getNamedAttr("device_tag", GetBuilder().getStringAttr(op.device_tag())));
  }
  attr_vec.push_back(GetBuilder().getNamedAttr(
      "scope_symbol_id", GetBuilder().getI64IntegerAttr(op.scope_symbol_id())));
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
  const ::oneflow::UserOpConf& user_conf = op.user_conf();
  const std::string& op_type_name = user_conf.op_type_name();

  std::vector<NamedAttribute> attr_vec;
  if (failed(AddOpConf(op, attr_vec))) { return failure(); }
  if (failed(AddDeviceName(op, attr_vec))) { return failure(); }
  attr_vec.push_back(GetBuilder().getNamedAttr(
      "op_type_name", GetBuilder().getStringAttr(op.user_conf().op_type_name())));
  std::vector<::mlir::Value> operand_vec;
  if (failed(namedAttributesFromUserOp(op, attr_vec))) { return failure(); }
  for (const auto& key : op.user_conf().input_order()) {
    auto& value = op.user_conf().input().at(key);
    int32_t index = 0;
    for (const std::string& lbn : value.s()) {
      if (failed(AppendDataInOperand(key, index, lbn, operand_vec))) { return failure(); }
      index += 1;
    }
  }
  if (failed(AppendCtrlInOperand(op, operand_vec))) { return failure(); }
  ::mlir::ValueRange operands(operand_vec);

  Operation* created_op = nullptr;

  auto out_types = llvm::SmallVector<Type, 8>();
  for (const auto& key : op.user_conf().output_order()) {
    auto& value = op.user_conf().output().at(key);
    for (const auto& output_lbn : value.s()) {
      out_types.push_back(GetTensorTypeOfLbn(output_lbn));
    }
  }
  if (op_type_name == "constant") {
    if (failed(AddOperandSegmentSizes(0, op.ctrl_in_op_name_size(), attr_vec))) {
      return failure();
    }
    ArrayRef<NamedAttribute> named_attributes(attr_vec);
    created_op = GetBuilder().create<oneflow::ConstantOp>(
        FileLineColLoc::get(GetMLIRContext(), op.name(), 0, 0), out_types, operands,
        named_attributes);
  } else {
    if (failed(AppendCtrlOutType(out_types))) { return failure(); }
    OperationState state(FileLineColLoc::get(GetMLIRContext(), op.name(), 0, 0), "oneflow.user");
    for (auto na : attr_vec) {
      if (na.first.str() == "input_lbn_segment_sizes") {
        int32_t data_input_size = 0;
        for (auto segment_size : na.second.dyn_cast<ArrayAttr>()) {
          data_input_size += segment_size.dyn_cast<IntegerAttr>().getInt();
        }
        if (failed(AddOperandSegmentSizes(data_input_size, op.ctrl_in_op_name_size(), attr_vec))) {
          return failure();
        }
      }
      if (na.first.str() == "output_lbns") {
        if (failed(AddResultSegmentSizes(na.second.dyn_cast<ArrayAttr>().size(), attr_vec))) {
          return failure();
        }
        if (na.second.dyn_cast<ArrayAttr>().size() != out_types.size() - 1) {
          GetModule()->emitError("len(out_types) - 1 != len(output_lbns), op: " + op.name());
          return failure();
        }
      }
    }
    ArrayRef<NamedAttribute> named_attributes(attr_vec);
    state.addAttributes(named_attributes);
    state.addOperands(operands);
    state.addTypes(out_types);
    created_op = GetBuilder().createOperation(state);
  }

  if (created_op == nullptr) {
    GetModule()->emitError("fail to create " + op.user_conf().op_type_name()
                           + " op, name: " + op.name());
    return failure();
  }
  if (failed(InsertOpResults(op, created_op))) { return failure(); }

  return success();
}  // namespace

LogicalResult ConvertCtrlInputs(Operation* op, ::oneflow::OperatorConf& op_conf) {
  if (auto ctrl_ins = GetCtrlIntputOperands(op)) {
    for (auto ctrl_in : ctrl_ins.getValue()) {
      op_conf.add_ctrl_in_op_name(
          ctrl_in.getDefiningOp()->getAttrOfType<StringAttr>("op_name").getValue().str());
    }
  }
  return success();
}

LogicalResult ConvertUserOpInputs(Operation* op, oneflow::UserOpAdaptor& user_op_adaptor,
                                  ::oneflow::UserOpConf* user_conf) {
  const std::string op_name = user_op_adaptor.op_name().getValue().str();
  int32_t input_idx = 0;
  if (auto keys = user_op_adaptor.input_lbn_segment_keys()) {
    auto sizes = user_op_adaptor.input_lbn_segment_sizes();
    if (keys.size() != sizes.size()) {
      op->emitError() << "fail to convert op inputs, input_lbn_segment_keys != "
                         "input_lbn_segment_sizes, name: "
                             + op_name;
      return failure();
    };
    // every key
    for (auto tuple : llvm::zip(keys, sizes)) {
      auto input_key = std::get<0>(tuple).dyn_cast<StringAttr>().getValue().str();
      auto input_size = std::get<1>(tuple).dyn_cast<IntegerAttr>().getInt();
      // every input for one key
      for (int32_t i = 0; i < input_size; i++) {
        if (auto result = op->getOperand(input_idx).dyn_cast<mlir::OpResult>()) {
          const std::string output_lbn_in_source_op =
              result.getDefiningOp()
                  ->getAttrOfType<ArrayAttr>("output_lbns")[result.getResultNumber()]
                  .dyn_cast<StringAttr>()
                  .getValue()
                  .str();
          *((*user_conf->mutable_input())[input_key].mutable_s()->Add()) = output_lbn_in_source_op;
          input_idx += 1;
        } else {
          op->emitError() << "fail to convert MLIR result to protobuf, name: " + op_name;
          op->dump();
          return failure();
        }
      }
    }
  } else {
    op->emitError() << "fail to convert op inputs, name: " + op_name;
    return failure();
  }
  return success();
}

LogicalResult ConvertUserOpOutputs(Operation* op, oneflow::UserOpAdaptor& user_op_adaptor,
                                   ::oneflow::UserOpConf* user_conf) {
  int32_t output_key_idx = -1;
  int32_t segment_offset = 0;
  for (const auto& result_and_idx : llvm::enumerate(GetDataOutputResults(op))) {
    const size_t result_idx = result_and_idx.index();
    if (result_idx == segment_offset) {
      output_key_idx += 1;
      int32_t size = user_op_adaptor.output_lbn_segment_sizes()[output_key_idx]
                         .dyn_cast<IntegerAttr>()
                         .getInt();
      segment_offset += size;
    }
    const std::string& output_key = user_op_adaptor.output_lbn_segment_keys()[output_key_idx]
                                        .dyn_cast<StringAttr>()
                                        .getValue()
                                        .str();
    const std::string& output_lbn =
        user_op_adaptor.output_lbns()[result_idx].dyn_cast<StringAttr>().getValue().str();
    *((*user_conf->mutable_output())[output_key].mutable_s()->Add()) = output_lbn;
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
  const std::string op_name = op->getAttrOfType<StringAttr>("op_name").getValue().str();
  for (auto id_attr : op->getAttrDictionary()) {
    auto id = id_attr.first;
    // mlir only attrs
    // TODO: find a way to skip attrs like callee in a declarative way
    if (id.strref().equals("callee") || id.strref().equals("device_name")
        || id.strref().equals("hierarchy") || id.strref().equals("input_lbn_segment_sizes")
        || id.strref().equals("output_lbns") || id.strref().equals("output_lbn_segment_sizes")
        || id.strref().equals("operand_segment_sizes")
        || id.strref().equals("result_segment_sizes")) {
      continue;
    }
    // convert op conf attributes
    else if (id.strref().equals("op_name")) {
      op_conf.set_name(op_name);
    } else if (id.strref().equals("op_type_name")) {
      user_conf->set_op_type_name(user_op_adaptor.op_type_name().getValue().str());
    } else if (id.strref().equals("device_tag")) {
      op_conf.set_device_tag(user_op_adaptor.device_tag().getValue().str());
    } else if (id.strref().equals("scope_symbol_id")) {
      op_conf.set_scope_symbol_id(user_op_adaptor.scope_symbol_id().getInt());
    } else if (id.strref().equals("input_lbn_segment_keys")) {
      for (auto s : user_op_adaptor.input_lbn_segment_keys().dyn_cast<ArrayAttr>().getValue()) {
        op_conf.mutable_user_conf()->add_input_order(s.dyn_cast<StringAttr>().getValue().str());
      }
    } else if (id.strref().equals("output_lbn_segment_keys")) {
      for (auto s : user_op_adaptor.output_lbn_segment_keys().dyn_cast<ArrayAttr>().getValue()) {
        op_conf.mutable_user_conf()->add_output_order(s.dyn_cast<StringAttr>().getValue().str());
      }
    }
    // convert user conf attributes
    else {
      auto attr_name = id.str();
      Attribute attr = id_attr.second;
      auto user_attr = ::oneflow::AttrValue();
      ::oneflow::AttrType attr_type =
          QueryAttrType(user_op_adaptor.op_type_name().getValue().str(), attr_name);
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
          for (auto int_v : s.dyn_cast<DenseIntElementsAttr>().getIntValues()) {
            assert(int_v.isSignedIntN(64));
            shape_ptr->mutable_dim()->Add(int_v.getSExtValue());
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

std::pair<unsigned, unsigned> getODSResultIndexAndLength(Operation* op, unsigned index) {
  auto sizeAttr = op->getAttrOfType<::mlir::DenseIntElementsAttr>("result_segment_sizes");

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i) start += (*(sizeAttr.begin() + i)).getZExtValue();
  unsigned size = (*(sizeAttr.begin() + index)).getZExtValue();
  return {start, size};
}

::mlir::Operation::result_range getODSResults(Operation* op, unsigned index) {
  auto valueRange = getODSResultIndexAndLength(op, index);
  return {std::next(op->result_begin(), valueRange.first),
          std::next(op->result_begin(), valueRange.first + valueRange.second)};
}

llvm::Optional<OpResult> GetCtrlOutputResult(Operation* op) {
  if (op->hasAttrOfType<::mlir::DenseIntElementsAttr>("result_segment_sizes")) {
    auto ctrl_output_result = getODSResults(op, 1);
    if (ctrl_output_result.empty()) {
      return llvm::None;
    } else {
      assert(ctrl_output_result.size() == 1);
      return ctrl_output_result.back();
    }
  } else {
    return llvm::None;
  }
}

ResultRange GetDataOutputResults(Operation* op) {
  if (op->hasAttrOfType<::mlir::DenseIntElementsAttr>("result_segment_sizes")) {
    return getODSResults(op, 0);
  } else {
    return op->getOpResults();
  }
}

}  // namespace mlir
