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
#include "OneFlow/UserOpConversion.h"
#include "oneflow/core/common/data_type.pb.h"
#include "oneflow/core/framework/user_op_conf.pb.h"
#include "oneflow/core/job/job.pb.h"
#include "oneflow/core/operator/op_conf.pb.h"
#include "oneflow/core/framework/user_op_def.h"
#include "oneflow/core/framework/user_op_registry_manager.h"

#include "OneFlow/OneFlowDialect.h"
#include "OneFlow/SBP/SBPDialect.h"
#include "OneFlow/SBP/SBPAttributes.h"
#include "OneFlow/OneFlowOps.h"
#include "OneFlow/UserOpReflection.h"
#include "OneFlow/OneFlowTypes.h"
#include "OneFlow/OneFlowSupport.h"
#include "OneFlow/Passes.h"
#include "OneFlow/MLIROneFlowTranslation.h"
#include "OneFlow/OneFlowSupport.h"
#include "OneFlow/OneFlowDataTypeConversion.h"

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

#include "oneflow/core/framework/sbp_context.h"
#include "oneflow/core/job/sbp_signature_builder.h"
namespace mlir {

namespace oneflow {

using PbMessage = google::protobuf::Message;

namespace {

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

LogicalResult IsAttrBelong2Op(const std::string& op_type_name, const std::string& attr_name) {
  ::oneflow::user_op::UserOpDefWrapper op_def(support::getUserOpDef(op_type_name));
  return success(op_def.IsAttrName(attr_name));
}

LogicalResult Importer::AddUserOpInputOutputSegments(const ::oneflow::OperatorConf& op,
                                                     std::vector<NamedAttribute>& attr_vec) {
  if (op.has_user_conf() == false) return failure();
  const auto& user_conf = op.user_conf();
  const ::oneflow::UserOpDef& op_def = support::getUserOpDef(op.user_conf().op_type_name());
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

ArrayAttr Importer::GetAttrFromStride(const ::oneflow::Int64ListProto& stride) {
  return GetBuilder().getArrayAttr(llvm::to_vector<8>(llvm::map_range(
      stride.dim(), [this](int64_t v) -> Attribute { return getSI64IntegerAttr(v); })));
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
    else if (value.has_at_stride()) {
      attr_vec.emplace_back(GetBuilder().getNamedAttr(name, GetAttrFromStride(value.at_stride())));
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
    else if (value.has_at_list_stride()) {
      auto dense_attr_list =
          llvm::map_range(value.at_list_stride().val(),
                          [&](const ::oneflow::Int64ListProto& s) { return GetAttrFromStride(s); });
      std::vector<mlir::Attribute> dense_attr_vector{dense_attr_list.begin(),
                                                     dense_attr_list.end()};
      attr_vec.emplace_back(
          GetBuilder().getNamedAttr(name, GetBuilder().getArrayAttr(dense_attr_vector)));
    }
    else if (value.has_at_complex_double()) {
      std::vector<mlir::Attribute> dense_attr_vector{
          GetBuilder().getF64FloatAttr(value.at_complex_double().real()),
          GetBuilder().getF64FloatAttr(value.at_complex_double().imag())};
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
      llvm::errs() << "unsupported sbp: " << nd_sbp.DebugString();
      exit(EXIT_FAILURE);
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
  const auto& op_def = support::getUserOpDef(op.user_conf().op_type_name());
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
  SetOpStateLoc(op, state);
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
          OpTrait::IsOpConfCompatible<void>::getOpName(ctrl_in.getDefiningOp()).str());
    }
  }
  return success();
}

LogicalResult ConvertUserOpInputs(Operation* op, StringRef op_name,
                                  ::oneflow::UserOpConf* user_conf) {
  std::vector<std::string> keys{};
  std::vector<int32_t> sizes{};
  if (failed(user_op::GetFilteredSegmentKeyAndSizes<OpTrait::AttrSizedOperandSegments>(op, keys,
                                                                                       sizes))) {
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
        *(input_s_ptr) = user_op::GetOutputLbn(result).getValue();
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
  if (failed(user_op::GetFilteredSegmentKeyAndSizes<OpTrait::AttrSizedResultSegments>(op, keys,
                                                                                      sizes))) {
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

void Importer::SetOpStateLoc(const ::oneflow::OperatorConf& op_conf, OperationState& state) {
  if (op_conf.has_loc()) {
    state.location = (FileLineColLoc::get(GetMLIRContext(), op_conf.loc(), 0, 0));
  }
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
    *var_op_conf->mutable_shape() = user_op::getAttrAsShape(shape_attr);
  }

  if (op->hasAttr(OpTrait::TensorSource<void>::getDataTypeAttrName())) {
    if (auto dt_mlir = op.data_type()) {
      const auto dt = support::FromMLIRDataTypeToOFDataType(dt_mlir.getValue());
      if (failed(dt)) { return failure(); }
      var_op_conf->set_data_type(dt.getValue());
    }
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

  for (auto output : op.parallel()->getOutputs()) {
    if (auto nd_outputs = output.dyn_cast<ArrayAttr>()) {
      for (auto nd_output : nd_outputs) {
        std::string sbp{};
        if (failed(SBPTranslation::PrintSbpAttrToString(nd_output, sbp))) return failure();
        var_op_conf->add_nd_sbp(sbp);
      }
    } else {
      std::string sbp{};
      if (failed(SBPTranslation::PrintSbpAttrToString(output, sbp))) return failure();
      var_op_conf->add_nd_sbp(sbp);
    }
  }
  // all operands are ctrl_inputs
  for (const auto& operand : op->getOperands()) {
    op_conf->add_ctrl_in_op_name(
        OpTrait::IsOpConfCompatible<void>::getOpName(operand.getDefiningOp()).str());
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
    *input_op_conf->mutable_blob_conf()->mutable_shape() = user_op::getAttrAsShape(shape_attr);
  }

  if (op->hasAttr(OpTrait::TensorSource<void>::getDataTypeAttrName())) {
    if (auto dt_mlir = op.data_type()) {
      const auto dt = support::FromMLIRDataTypeToOFDataType(dt_mlir.getValue());
      if (failed(dt)) { return failure(); }
      input_op_conf->mutable_blob_conf()->set_data_type(dt.getValue());
    }
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
        OpTrait::IsOpConfCompatible<void>::getOpName(op->getOperand(i).getDefiningOp()).str());
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
    *output_op_conf->mutable_blob_conf()->mutable_shape() = user_op::getAttrAsShape(shape_attr);
  }

  if (op->hasAttr(OpTrait::TensorSource<void>::getDataTypeAttrName())) {
    if (auto dt_mlir = op.data_type()) {
      const auto dt = support::FromMLIRDataTypeToOFDataType(dt_mlir.getValue());
      if (failed(dt)) { return failure(); }
      output_op_conf->mutable_blob_conf()->set_data_type(dt.getValue());
    }
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
  auto output_lbn = user_op::GetOutputLbn(result).getValue();
  output_op_conf->set_in(output_lbn);
  for (size_t i = 1; i < op->getNumOperands(); ++i) {
    op_conf->add_ctrl_in_op_name(
        OpTrait::IsOpConfCompatible<void>::getOpName(op->getOperand(i).getDefiningOp()).str());
  }
  return success();
}

}  // namespace oneflow

}  // namespace mlir
