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
#include "oneflow/core/common/data_type.pb.h"
#include "oneflow/core/framework/user_op_conf.pb.h"
#include "oneflow/core/job/job.pb.h"
#include "oneflow/core/operator/op_conf.pb.h"
#include "oneflow/core/framework/user_op_def.h"
#include "oneflow/core/framework/user_op_registry_manager.h"

#include "OneFlow/OneFlowDialect.h"
#include "OneFlow/OneFlowOps.h"
#include "OneFlow/OneFlowTypes.h"
#include "OneFlow/OneFlowSupport.h"
#include "OneFlow/Passes.h"
#include "OneFlow/MLIROneFlowTranslation.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/OperationSupport.h"

#include "mlir/IR/UseDefLists.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Tools/mlir-translate/MlirTranslateMain.h"
#include "mlir/Tools/mlir-translate/Translation.h"
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

#include <google/protobuf/text_format.h>

namespace mlir {

namespace oneflow {

using PbMessage = google::protobuf::Message;

namespace {

const ::oneflow::UserOpDef& GetUserOpDef(const std::string& op_type_name) {
  const ::oneflow::user_op::OpRegistryResult* val =
      ::oneflow::user_op::UserOpRegistryMgr::Get().GetOpRegistryResult(op_type_name);
  CHECK(val) << " Cannot find op_type_name: " << op_type_name;
  return val->op_def;
}

::oneflow::AttrType QueryAttrType(const std::string& op_type_name, const std::string& attr_name) {
  ::oneflow::user_op::UserOpDefWrapper op_def(GetUserOpDef(op_type_name));
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
    const auto& it = op.user_conf().output().find(key);
    if (it == op.user_conf().output().end()) { continue; }
    auto result_size = it->second.s_size();
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
  const auto UserOpOperationName = OperationName(UserOp::getOperationName(), GetMLIRContext());
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

llvm::Optional<mlir::oneflow::DataTypeAttr> GetDataTypeAttr(MLIRContext* context,
                                                            ::oneflow::DataType oneflow_value) {
  switch (oneflow_value) {
    case ::oneflow::DataType::kInvalidDataType:
      return oneflow::DataTypeAttr::get(context, mlir::oneflow::DataType::DT_InvalidDataType);
      break;
#define DEFINE_ONE_ELIF(datatype)                                                       \
  case ::oneflow::DataType::k##datatype:                                                \
    return oneflow::DataTypeAttr::get(context, mlir::oneflow::DataType::DT_##datatype); \
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
      DEFINE_ONE_ELIF(BFloat16)
      DEFINE_ONE_ELIF(Bool)
#undef DEFINE_ONE_ELIF
    default: llvm::errs() << "unsupported data type: " << oneflow_value << "\n"; return llvm::None;
  }
}

ArrayAttr Importer::GetAttrFromShape(const ::oneflow::ShapeProto& shape) {
  return GetBuilder().getArrayAttr(llvm::to_vector<8>(llvm::map_range(
      shape.dim(), [this](int64_t v) -> Attribute { return getSI64IntegerAttr(v); })));
}

void WriteAttrToShape(mlir::Attribute& attr, ::oneflow::ShapeProto* shape) {
  for (auto v : attr.dyn_cast<ArrayAttr>().getValue()) {
    shape->add_dim(v.dyn_cast<IntegerAttr>().getSInt());
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
      mlir::NamedAttribute kv =
          GetBuilder().getNamedAttr(name, GetBuilder().getSI32IntegerAttr(value.at_int32()));
      attr_vec.emplace_back(kv);
    } else if (value.has_at_int64()) {
      mlir::NamedAttribute kv =
          GetBuilder().getNamedAttr(name, getSI64IntegerAttr(value.at_int64()));
      attr_vec.emplace_back(kv);
    }
#define DEFINE_ONE_ELIF(at_key, get_attr)                                       \
  else if (value.has_##at_key()) {                                              \
    mlir::NamedAttribute kv =                                                   \
        GetBuilder().getNamedAttr(name, GetBuilder().get_attr(value.at_key())); \
    attr_vec.emplace_back(kv);                                                  \
  }
    DEFINE_ONE_ELIF(at_bool, getBoolAttr)
    DEFINE_ONE_ELIF(at_float, getF32FloatAttr)
    DEFINE_ONE_ELIF(at_double, getF64FloatAttr)
    DEFINE_ONE_ELIF(at_string, getStringAttr)
#undef DEFINE_ONE_ELIF
    else if (value.has_at_shape()) {
      attr_vec.emplace_back(GetBuilder().getNamedAttr(name, GetAttrFromShape(value.at_shape())));
    }
#define DEFINE_ONE_ELIF(at_key, get_attr, field)                                         \
  else if (value.has_##at_key()) {                                                       \
    mlir::NamedAttribute kv = GetBuilder().getNamedAttr(                                 \
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
      mlir::NamedAttribute kv =
          GetBuilder().getNamedAttr(name, GetBuilder().getStrArrayAttr(r_vec));
      attr_vec.emplace_back(kv);
    }
    else if (value.has_at_data_type()) {
      if (auto dt_attr = GetDataTypeAttr(GetMLIRContext(), value.at_data_type())) {
        mlir::NamedAttribute kv = GetBuilder().getNamedAttr(name, dt_attr.getValue());
        attr_vec.emplace_back(kv);
      } else {
        GetModule().emitError("fail to convert op attr, key: " + name);
        return failure();
      }
    }
    else if (value.has_at_list_data_type()) {
      auto dt_attr_list =
          llvm::map_range(value.at_list_data_type().val(), [&](auto t) -> mlir::Attribute {
            auto dt = GetDataTypeAttr(GetMLIRContext(), static_cast<::oneflow::DataType>(t));
            CHECK(dt) << "fail to convert op attr, key: " + name;
            return dt.getValue();
          });
      attr_vec.emplace_back(GetBuilder().getNamedAttr(
          name, GetBuilder().getArrayAttr(llvm::to_vector<8>(dt_attr_list))));
    }
    else if (value.has_at_list_shape()) {
      auto dense_attr_list =
          llvm::map_range(value.at_list_shape().val(),
                          [&](const ::oneflow::ShapeProto& s) { return GetAttrFromShape(s); });
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
    if (dt == ::oneflow::DataType::kInvalidDataType) {
      return InvalidElementType::get(GetMLIRContext());
    }
    if (dt == ::oneflow::DataType::kChar) { return CharElementType::get(GetMLIRContext()); }
    if (dt == ::oneflow::DataType::kFloat) { return GetBuilder().getF32Type(); }
    if (dt == ::oneflow::DataType::kDouble) { return GetBuilder().getF64Type(); }
    if (dt == ::oneflow::DataType::kInt8) { return GetBuilder().getIntegerType(8, true); }
    if (dt == ::oneflow::DataType::kInt32) { return GetBuilder().getI32Type(); }
    if (dt == ::oneflow::DataType::kInt64) { return GetBuilder().getI64Type(); }
    if (dt == ::oneflow::DataType::kUInt8) { return GetBuilder().getIntegerType(8, false); }
    if (dt == ::oneflow::DataType::kOFRecord) { return OFRecordElementType::get(GetMLIRContext()); }
    if (dt == ::oneflow::DataType::kFloat16) { return GetBuilder().getF16Type(); }
    if (dt == ::oneflow::DataType::kTensorBuffer) {
      return TensorBufferElementType::get(GetMLIRContext());
    }
    if (dt == ::oneflow::DataType::kBool) { return GetBuilder().getI8Type(); }
    if (dt == ::oneflow::DataType::kUInt16) { return GetBuilder().getIntegerType(16, false); }
    if (dt == ::oneflow::DataType::kUInt32) { return GetBuilder().getI32Type(); }
    if (dt == ::oneflow::DataType::kUInt64) { return GetBuilder().getI64Type(); }
    if (dt == ::oneflow::DataType::kUInt128) { return GetBuilder().getIntegerType(128, false); }
    llvm::errs() << "unsupported data type: " << dt << "\n";
    return llvm::None;
  }
}

LogicalResult ParseNdSbpFromAttr(::llvm::ArrayRef<Attribute> nd_sbp_attr,
                                 ::oneflow::NdSbp* nd_sbp) {
  for (const auto& sbp_attr : nd_sbp_attr) {
    auto sbp_str_attr = sbp_attr.dyn_cast<StringAttr>();
    if (!sbp_str_attr) {
      llvm::errs() << "nd_sbp attr is not a StrArrayAttr";
      return failure();
    }
    auto sbp_strref = sbp_str_attr.getValue();
    if (sbp_strref.startswith("S")) {
      if (!(sbp_strref.substr(1, 1) == "(" && sbp_strref.endswith(")"))) {
        llvm::errs() << "invalid sbp S(x) string value: " << sbp_strref;
        return failure();
      }
      auto split_axis = std::stoi(sbp_strref.substr(2, 1).str());
      nd_sbp->add_sbp_parallel()->mutable_split_parallel()->set_axis(split_axis);
    } else if (sbp_strref == "B") {
      nd_sbp->add_sbp_parallel()->mutable_broadcast_parallel();
    } else if (sbp_strref == "P") {
      nd_sbp->add_sbp_parallel()->mutable_partial_sum_parallel();
    } else {
      llvm::errs() << "unspported nd_sbp string value: " << sbp_strref;
      return failure();
    }
  }
  return success();
}

Attribute ConvertNdSbpToAttr(Builder& builder, const ::oneflow::NdSbp& nd_sbp) {
  llvm::SmallVector<std::string, 2> sbp_strs;
  for (const auto& sbp : nd_sbp.sbp_parallel()) {
    if (sbp.has_split_parallel()) {
      sbp_strs.emplace_back("S(" + std::to_string(sbp.split_parallel().axis()) + ")");
    } else if (sbp.has_broadcast_parallel()) {
      sbp_strs.emplace_back("B");
    } else if (sbp.has_partial_sum_parallel()) {
      sbp_strs.emplace_back("P");
    } else {
      llvm::errs() << "unsupported sbp";
    }
  }
  return builder.getStrArrayAttr(
      makeArrayRef(llvm::SmallVector<StringRef>(sbp_strs.begin(), sbp_strs.end())));
}

LogicalResult ValidateUserOpConf(const ::oneflow::OperatorConf& op_conf, UserOpArgs args,
                                 UserOpArgDefs arg_defs) {
  for (const auto& input_arg : args) {
    const bool found = std::find_if(arg_defs.begin(), arg_defs.end(),
                                    [&](const ::oneflow::UserOpDef_ArgDef& arg_def) {
                                      return input_arg.first == arg_def.name();
                                    })
                       != arg_defs.end();
    if (!found) {
      llvm::errs() << "fail to validate user op conf, arg def of arg not found: " << input_arg.first
                   << ", op: \n"
                   << op_conf.DebugString() << "\n";
      return failure();
    }
  }
  return success();
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
  if (failed(ValidateUserOpConf(op, op.user_conf().input(), op_def.input()))) { return failure(); }
  if (failed(ValidateUserOpConf(op, op.user_conf().output(), op_def.output()))) {
    return failure();
  }
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
  OperationState state(FileLineColLoc::get(GetMLIRContext(), op.name(), 0, 0),
                       UserOp::getOperationName());
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
  created_op = GetBuilder().create(state);

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
    if (failed(GetFilteredSegmentKeyAndSizes<OpTrait::AttrSizedResultSegments>(def_op, def_op_keys,
                                                                               def_op_sizes))) {
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
        return def_op->getAttrOfType<StringAttr>(OpTrait::IsOpConfCompatible<void>::getOpNameAttr())
                   .str()
               + "/" + name + "_" + std::to_string(bn_i);
      }
      size_sum += size;
    }
  }
  return llvm::None;
}

LogicalResult ConvertUserOpInputs(Operation* op, StringRef op_name,
                                  ::oneflow::UserOpConf* user_conf) {
  std::vector<std::string> keys{};
  std::vector<int32_t> sizes{};
  if (failed(GetFilteredSegmentKeyAndSizes<OpTrait::AttrSizedOperandSegments>(op, keys, sizes))) {
    op->emitError("fail to convert user op inputs");
    return failure();
  }
  int32_t input_idx = 0;
  for (auto tuple : llvm::zip(keys, sizes)) {
    auto input_key = std::get<0>(tuple);
    auto input_size = std::get<1>(tuple);
    if (input_size <= 0)
      return op->emitError("input_size <= 0, op: " + op->getName().getStringRef());
    for (int32_t i = 0; i < input_size; i++) {
      if (auto result = GetDataInputOperands(op)[input_idx].dyn_cast<mlir::OpResult>()) {
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

LogicalResult ConvertUserOpOutputs(Operation* op, StringRef op_name,
                                   ::oneflow::UserOpConf* user_conf) {
  std::vector<std::string> keys{};
  std::vector<int32_t> sizes{};
  if (failed(GetFilteredSegmentKeyAndSizes<OpTrait::AttrSizedResultSegments>(op, keys, sizes))) {
    op->emitError("fail to convert user op outputs");
    return failure();
  }
  for (auto tuple : llvm::zip(keys, sizes)) {
    auto name = std::get<0>(tuple);
    auto result_size = std::get<1>(tuple);
    if (result_size == 0) continue;
    for (int32_t i = 0; i < result_size; i++) {
      auto out_s_ptr = (*user_conf->mutable_output())[name].mutable_s()->Add();
      *(out_s_ptr) = op_name.str() + "/" + name + "_" + std::to_string(i);
    }
  }
  return success();
}

LogicalResult ConvertDT(::mlir::oneflow::DataType data_type_mlir, ::oneflow::DataType& data_type) {
  switch (data_type_mlir) {
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
      DEFINE_ONE_CASE(Bool)
#undef DEFINE_ONE_CASE
    default: return failure();
  }
  return success();
}

LogicalResult ConvertDTFromAttr(Attribute attr, ::oneflow::DataType& data_type) {
  auto dt_attr = attr.dyn_cast<mlir::oneflow::DataTypeAttr>();
  return ConvertDT(dt_attr.getValue(), data_type);
}

LogicalResult Importer::ConvertUserOpAttributes(Operation* op, ::oneflow::OperatorConf& op_conf) {
  auto user_conf = op_conf.mutable_user_conf();
  std::string op_type_name = GetOpTypeName(op);
  op_conf.mutable_user_conf()->set_op_type_name(op_type_name);
  if (op->hasTrait<OpTrait::IsOpConfCompatible>()) {
    if (OpTrait::IsOpConfCompatible<void>::dump_attr(op, &op_conf).failed()) {
      return op->emitError("fail to save attr to op_conf");
    }
  }

  for (auto id_attr : op->getAttrDictionary()) {
    auto id = id_attr.getName();
    // mlir only attrs
    // TODO: find a way to skip attrs like callee in a declarative way
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
    } else if (id.strref().equals("input_sizes") || id.strref().equals("output_sizes")) {
      continue;
    }
    // convert op conf attributes
    else if (id.strref().equals(OpTrait::IsOpConfCompatible<void>::getOpNameAttr())) {
      continue;
    } else if (id.strref().equals(OpTrait::IsOpConfCompatible<void>::getDeviceTagAttr())) {
      continue;
    } else if (id.strref().equals(OpTrait::IsOpConfCompatible<void>::getScopeSymbolIDAttr())) {
      continue;
    }
    // convert user conf attributes
    else {
      auto attr_name = id.str();
      Attribute attr = id_attr.getValue();
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
        WriteAttrToShape(attr, user_attr.mutable_at_shape());
      } else if (attr_type == ::oneflow::kAtDataType) {
        ::oneflow::DataType dt = ::oneflow::kInvalidDataType;
        if (succeeded(ConvertDTFromAttr(attr, dt))) {
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
          if (succeeded(ConvertDTFromAttr(v, dt))) {
            user_attr.mutable_at_list_data_type()->add_val(dt);
          } else {
            op->emitError() << "fail to convert op attr to data type, key: " + id.str();
            return failure();
          }
        }
      } else if (attr_type == ::oneflow::kAtListShape) {
        for (auto shape_attr : attr.dyn_cast<ArrayAttr>().getValue()) {
          ::oneflow::ShapeProto* shape_ptr = user_attr.mutable_at_list_shape()->add_val();
          WriteAttrToShape(shape_attr, shape_ptr);
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
  {
    std::vector<std::string> keys{};
    std::vector<int32_t> sizes{};
    if (failed(GetFilteredSegmentKeyAndSizes<OpTrait::AttrSizedOperandSegments>(op, keys, sizes))) {
      op->emitError("fail to convert user op input order");
      return failure();
    }
    for (const auto& s : keys) { op_conf.mutable_user_conf()->add_input_order(s); }
  }
  {
    std::vector<std::string> keys{};
    std::vector<int32_t> sizes{};
    if (failed(GetFilteredSegmentKeyAndSizes<OpTrait::AttrSizedResultSegments>(op, keys, sizes))) {
      op->emitError("fail to convert user op output order");
      return failure();
    }
    for (const auto& s : keys) { op_conf.mutable_user_conf()->add_output_order(s); }
  }
  return success();
}

LogicalResult ConvertVariableOpConf(VariableOp op, ::oneflow::OperatorConf* op_conf) {
  op_conf->set_name(op.op_name().str());
  op_conf->set_device_tag(op.device_tag().str());
  if (auto scope_symbol_id = op.scope_symbol_id()) {
    op_conf->set_scope_symbol_id(scope_symbol_id.getValue());
  }
  // TODO: process stream_name_hint

  auto* var_op_conf = op_conf->mutable_variable_conf();
  var_op_conf->set_out("out");

  if (auto shape_attr =
          op->getAttrOfType<ArrayAttr>(OpTrait::TensorSource<void>::getShapeAttrName())) {
    WriteAttrToShape(shape_attr, var_op_conf->mutable_shape());
  }

  if (op->hasAttr(OpTrait::TensorSource<void>::getDataTypeAttrName())) {
    ::oneflow::DataType dt = ::oneflow::DataType::kInvalidDataType;
    if (auto dt_mlir = op.data_type()) {
      if (failed(ConvertDT(dt_mlir.getValue(), dt))) { return failure(); }
    }
    var_op_conf->set_data_type(dt);
  }

  if (op->hasAttr("model_name")) { var_op_conf->set_model_name(op.model_name().str()); }

  if (op->hasAttr("l1_regularization")) {
    var_op_conf->mutable_regularizer()->mutable_l1_l2_conf()->set_l1(
        op.l1_regularization().convertToFloat());
  }

  if (op->hasAttr("l2_regularization")) {
    var_op_conf->mutable_regularizer()->mutable_l1_l2_conf()->set_l2(
        op.l2_regularization().convertToFloat());
  }

  if (op->hasAttr("trainable")) { var_op_conf->set_trainable(op.trainable()); }

  for (const auto& sbp : op.nd_sbp()) {
    var_op_conf->add_nd_sbp(sbp.cast<StringAttr>().getValue().str());
  }

  // all operands are ctrl_inputs
  for (const auto& operand : op->getOperands()) {
    op_conf->add_ctrl_in_op_name(
        operand.getDefiningOp()->getAttrOfType<StringAttr>("op_name").getValue().str());
  }
  if (auto floatInit = op.float_initializer()) {
    var_op_conf->mutable_initializer()->mutable_constant_conf()->set_value(
        floatInit.getValue().convertToFloat());
  } else if (auto integerInit = op.integer_initializer()) {
    var_op_conf->mutable_initializer()->mutable_constant_int_conf()->set_value(
        integerInit.getValue());
  } else {
    // empty initializer
    var_op_conf->mutable_initializer()->mutable_empty_conf();
  }

  return success();
}

LogicalResult ConvertInputOpConf(InputOp op, ::oneflow::OperatorConf* op_conf) {
  op_conf->set_name(op.op_name().str());
  op_conf->set_device_tag(op.device_tag().str());
  if (auto scope_symbol_id = op.scope_symbol_id()) {
    op_conf->set_scope_symbol_id(scope_symbol_id.getValue());
  }
  // TODO: process stream_name_hint

  auto* input_op_conf = op_conf->mutable_input_conf();
  input_op_conf->set_out("out");

  if (auto shape_attr =
          op->getAttrOfType<ArrayAttr>(OpTrait::TensorSource<void>::getShapeAttrName())) {
    WriteAttrToShape(shape_attr, input_op_conf->mutable_blob_conf()->mutable_shape());
  }

  if (op->hasAttr(OpTrait::TensorSource<void>::getDataTypeAttrName())) {
    ::oneflow::DataType dt = ::oneflow::DataType::kInvalidDataType;
    if (auto dt_mlir = op.data_type()) {
      if (failed(ConvertDT(dt_mlir.getValue(), dt))) { return failure(); }
    }
    input_op_conf->mutable_blob_conf()->set_data_type(dt);
  }

  if (op->hasAttr(OpTrait::TensorSource<void>::getIsDynamicAttrName())) {
    input_op_conf->mutable_blob_conf()->set_is_dynamic(op.is_dynamic().getValue());
  }

  if (op->hasAttr(OpTrait::TensorSource<void>::getNdSbpAttrName())) {
    if (failed(ParseNdSbpFromAttr(op.nd_sbp()->getValue(),
                                  input_op_conf->mutable_blob_conf()->mutable_nd_sbp()))) {
      return failure();
    }
  }

  if (op->hasAttr("job_name")) { input_op_conf->set_job_name(op.job_name().getValue().str()); }

  // operand 0 is block argument, others are ctrl_inputs
  for (size_t i = 1; i < op->getNumOperands(); ++i) {
    op_conf->add_ctrl_in_op_name(
        op->getOperand(i).getDefiningOp()->getAttrOfType<StringAttr>("op_name").getValue().str());
  }

  return success();
}

LogicalResult ConvertOutputOpConf(OutputOp op, ::oneflow::OperatorConf* op_conf) {
  op_conf->set_name(op.op_name().str());
  op_conf->set_device_tag(op.device_tag().str());
  if (auto scope_symbol_id = op.scope_symbol_id()) {
    op_conf->set_scope_symbol_id(scope_symbol_id.getValue());
  }
  // TODO: process stream_name_hint

  auto* output_op_conf = op_conf->mutable_output_conf();
  output_op_conf->set_out("out");

  if (auto shape_attr =
          op->getAttrOfType<ArrayAttr>(OpTrait::TensorSource<void>::getShapeAttrName())) {
    WriteAttrToShape(shape_attr, output_op_conf->mutable_blob_conf()->mutable_shape());
  }

  if (op->hasAttr(OpTrait::TensorSource<void>::getDataTypeAttrName())) {
    ::oneflow::DataType dt = ::oneflow::DataType::kInvalidDataType;
    if (auto dt_mlir = op.data_type()) {
      if (failed(ConvertDT(dt_mlir.getValue(), dt))) { return failure(); }
    }
    output_op_conf->mutable_blob_conf()->set_data_type(dt);
  }

  if (op->hasAttr(OpTrait::TensorSource<void>::getIsDynamicAttrName())) {
    output_op_conf->mutable_blob_conf()->set_is_dynamic(op.is_dynamic().getValue());
  }

  if (op->hasAttr(OpTrait::TensorSource<void>::getNdSbpAttrName())) {
    if (failed(ParseNdSbpFromAttr(op.nd_sbp()->getValue(),
                                  output_op_conf->mutable_blob_conf()->mutable_nd_sbp()))) {
      return failure();
    }
  }

  if (op->hasAttr("job_name")) { output_op_conf->set_job_name(op.job_name().getValue().str()); }

  if (op->getNumOperands() == 0) {
    op->emitError("output op has at least one input.");
    return failure();
  }
  auto result = op->getOperand(0).dyn_cast<mlir::OpResult>();
  auto output_lbn = GetOutputLbn(result).getValue();
  output_op_conf->set_in(output_lbn);
  for (size_t i = 1; i < op->getNumOperands(); ++i) {
    op_conf->add_ctrl_in_op_name(
        op->getOperand(i).getDefiningOp()->getAttrOfType<StringAttr>("op_name").getValue().str());
  }
  return success();
}

}  // namespace oneflow

}  // namespace mlir
