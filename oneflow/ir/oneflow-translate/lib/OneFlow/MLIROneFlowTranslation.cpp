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
#include "OneFlow/OneFlowOps.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
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
#include "mlir/Translation.h"
#include "llvm-c/Core.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/None.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"

#include "OneFlow/OneFlowDialect.h"

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

#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "OneFlow/MLIROneFlowTranslation.h"

namespace mlir {

namespace {

using PbMessage = google::protobuf::Message;

class Importer {
 public:
  Importer(RoundTripOneFlowJobWrapperInterface& job_wrapper, MLIRContext* context, ModuleOp module)
      : builder_(context),
        context_(context),
        module_(module),
        unknown_loc_(FileLineColLoc::get("imported-protobuf", 0, 0, context)),
        job_(job_wrapper.job()),
        job_wrapper_(job_wrapper) {}

  LogicalResult namedAttributesFromUserOp(const ::oneflow::OperatorConf& op,
                                          std::vector<NamedAttribute>& attr_vec);
  LogicalResult AppendDataInOperand(const std::string& lbn,
                                    std::vector<::mlir::Value>& operand_vec);
  LogicalResult AppendCtrlInOperand(const ::oneflow::OperatorConf& op,
                                    std::vector<::mlir::Value>& operand_vec);
  LogicalResult AppendCtrlOutType(llvm::SmallVector<Type, 8>& out_types);
  LogicalResult AddOpConf(const ::oneflow::OperatorConf& op, std::vector<NamedAttribute>& attr_vec);
  LogicalResult AddUserOpInputOutputSegments(const ::oneflow::OperatorConf& op,
                                             std::vector<NamedAttribute>& attr_vec);
  LogicalResult AddPlacement(const ::oneflow::OperatorConf& op,
                             std::vector<NamedAttribute>& attr_vec);
  LogicalResult AddOperandSegmentSizes(int input_lbns_size, int ctrl_in_size,
                                       std::vector<NamedAttribute>& attr_vec);
  LogicalResult AddResultSegmentSizes(int output_lbns_size, std::vector<NamedAttribute>& attr_vec);
  LogicalResult InsertOpResults(Operation*);
  LogicalResult ProcessUserOp(const ::oneflow::OperatorConf& op);
  LogicalResult ProcessSystemOp(const ::oneflow::OperatorConf& op);
  LogicalResult ProcessJob();
  LogicalResult TryToUpdateJob();

  DenseIntElementsAttr DenseIntElementsAttrFromShape(const ::oneflow::ShapeProto& shape);
  void ConvertUseropAttributes(Operation* op, ::oneflow::OperatorConf& op_conf,
                               std::string& err_str);

  IntegerAttr getSI64IntegerAttr(int64_t value) {
    return IntegerAttr::get(builder_.getIntegerType(64, /*isSigned=*/true),
                            APInt(64, value, /*isSigned=*/true));
  }
  ArrayAttr getSI32ArrayAttr(ArrayRef<int32_t> values) {
    auto attrs = llvm::to_vector<8>(llvm::map_range(
        values, [this](int32_t v) -> Attribute { return builder_.getSI32IntegerAttr(v); }));
    return builder_.getArrayAttr(attrs);
  }
  ArrayAttr getSI64ArrayAttr(ArrayRef<int64_t> values) {
    auto attrs = llvm::to_vector<8>(
        llvm::map_range(values, [this](int64_t v) -> Attribute { return getSI64IntegerAttr(v); }));
    return builder_.getArrayAttr(attrs);
  }

 private:
  /// The current builder, pointing at where the next Instruction should be
  /// generated.
  OpBuilder builder_;
  /// The current context.
  MLIRContext* context_;
  /// The current module being created.
  ModuleOp module_;
  /// Cached FileLineColLoc::get("imported-protobuf", 0, 0).
  Location unknown_loc_;
  std::unordered_map<std::string, mlir::OpResult> lbn2result_;
  std::unordered_map<std::string, mlir::OpResult> op_name2ctrl_result_;
  const ::oneflow::Job* job_;
  RoundTripOneFlowJobWrapperInterface& job_wrapper_;
};

LogicalResult Importer::AddUserOpInputOutputSegments(const ::oneflow::OperatorConf& op,
                                                     std::vector<NamedAttribute>& attr_vec) {
  std::vector<llvm::StringRef> input_lbn_segment_keys;
  std::vector<int> input_lbn_segment_sizes;
  int data_input_size = 0;
  for (auto input : op.user_conf().input()) {
    input_lbn_segment_keys.push_back(input.first);
    input_lbn_segment_sizes.push_back(input.second.s_size());
    data_input_size += input.second.s_size();
  }
  attr_vec.push_back(builder_.getNamedAttr("input_lbn_segment_keys",
                                           builder_.getStrArrayAttr(input_lbn_segment_keys)));
  attr_vec.push_back(builder_.getNamedAttr("input_lbn_segment_sizes",
                                           builder_.getI32ArrayAttr(input_lbn_segment_sizes)));

  std::vector<llvm::StringRef> output_lbns;
  std::vector<llvm::StringRef> output_lbn_segment_keys;
  std::vector<int> output_lbn_segment_sizes;
  int data_output_size = 0;
  for (auto output : op.user_conf().output()) {
    output_lbns.insert(output_lbns.end(), output.second.s().begin(), output.second.s().end());
    output_lbn_segment_keys.push_back(output.first);
    output_lbn_segment_sizes.push_back(output.second.s_size());
    data_output_size += output.second.s_size();
  }
  attr_vec.push_back(builder_.getNamedAttr("output_lbns", builder_.getStrArrayAttr(output_lbns)));
  attr_vec.push_back(builder_.getNamedAttr("output_lbn_segment_keys",
                                           builder_.getStrArrayAttr(output_lbn_segment_keys)));
  attr_vec.push_back(builder_.getNamedAttr("output_lbn_segment_sizes",
                                           builder_.getI32ArrayAttr(output_lbn_segment_sizes)));
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
                                              builder_.getIntegerType(64, true));
  ;
  return DenseIntElementsAttr::get(tt, values);
}

LogicalResult Importer::namedAttributesFromUserOp(const ::oneflow::OperatorConf& op,
                                                  std::vector<NamedAttribute>& attr_vec) {
  if (op.has_user_conf() == false) {
    module_.emitError("Not a user op. op name: " + op.name());
    return failure();
  }
  for (const google::protobuf::MapPair<class std::basic_string<char>, ::oneflow::AttrValue>& attr :
       op.user_conf().attr()) {
    const std::string& name = attr.first;
    const ::oneflow::AttrValue& value = attr.second;
    if (value.has_at_int32()) {
      std::pair<mlir::Identifier, mlir::Attribute> kv =
          builder_.getNamedAttr(name, builder_.getSI32IntegerAttr(value.at_int32()));
      attr_vec.emplace_back(kv);
    } else if (value.has_at_int64()) {
      std::pair<mlir::Identifier, mlir::Attribute> kv =
          builder_.getNamedAttr(name, getSI64IntegerAttr(value.at_int64()));
      attr_vec.emplace_back(kv);
    }
#define DEFINE_ONE_ELIF(at_key, get_attr)                               \
  else if (value.has_##at_key()) {                                      \
    std::pair<mlir::Identifier, mlir::Attribute> kv =                   \
        builder_.getNamedAttr(name, builder_.get_attr(value.at_key())); \
    attr_vec.emplace_back(kv);                                          \
  }
    DEFINE_ONE_ELIF(at_bool, getBoolAttr)
    DEFINE_ONE_ELIF(at_float, getF32FloatAttr)
    DEFINE_ONE_ELIF(at_double, getF64FloatAttr)
    DEFINE_ONE_ELIF(at_string, getStringAttr)
#undef DEFINE_ONE_ELIF
    else if (value.has_at_shape()) {
      attr_vec.emplace_back(
          builder_.getNamedAttr(name, DenseIntElementsAttrFromShape(value.at_shape())));
    }
#define DEFINE_ONE_ELIF(at_key, get_attr, field)                                         \
  else if (value.has_##at_key()) {                                                       \
    std::pair<mlir::Identifier, mlir::Attribute> kv = builder_.getNamedAttr(             \
        name, get_attr({value.at_key().field().begin(), value.at_key().field().end()})); \
    attr_vec.emplace_back(kv);                                                           \
  }
    DEFINE_ONE_ELIF(at_list_int32, getSI32ArrayAttr, val)
    DEFINE_ONE_ELIF(at_list_int64, getSI64ArrayAttr, val)
    DEFINE_ONE_ELIF(at_list_float, builder_.getF32ArrayAttr, val)
#undef DEFINE_ONE_ELIF
    else if (value.has_at_list_string()) {
      std::vector<llvm::StringRef> r_vec = {value.at_list_string().val().begin(),
                                            value.at_list_string().val().end()};
      std::pair<mlir::Identifier, mlir::Attribute> kv =
          builder_.getNamedAttr(name, builder_.getStrArrayAttr(r_vec));
      attr_vec.emplace_back(kv);
    }
    else if (value.has_at_data_type()) {
      std::string stringified = "";
      if (failed(StringifyDataType(value.at_data_type(), stringified))) {
        module_.emitError("fail to convert op attr, key: " + name);
        return failure();
      }
      std::pair<mlir::Identifier, mlir::Attribute> kv =
          builder_.getNamedAttr(name, builder_.getStringAttr(stringified));
      attr_vec.emplace_back(kv);
    }
    else if (value.has_at_list_data_type()) {
      auto stringified_list = llvm::map_range(value.at_list_data_type().val(), [&](int t) {
        std::string stringified = "";
        assert(succeeded(StringifyDataType(static_cast<::oneflow::DataType>(t), stringified)));
        return stringified;
      });
      std::vector<std::string> stringified_vector = {stringified_list.begin(),
                                                     stringified_list.end()};
      attr_vec.emplace_back(builder_.getNamedAttr(
          name, builder_.getStrArrayAttr(std::vector<StringRef>(
                    {stringified_vector.begin(), stringified_vector.end()}))));
    }
    else if (value.has_at_list_shape()) {
      auto dense_attr_list = llvm::map_range(
          value.at_list_shape().val(),
          [&](::oneflow::ShapeProto s) { return DenseIntElementsAttrFromShape(s); });
      std::vector<mlir::Attribute> dense_attr_vector{dense_attr_list.begin(),
                                                     dense_attr_list.end()};
      attr_vec.emplace_back(builder_.getNamedAttr(name, builder_.getArrayAttr(dense_attr_vector)));
    }
    else {
      module_.emitError("can't handle user op attr: " + name + ", op name: " + op.name()
                        + ", op type name: " + op.user_conf().op_type_name());
      return failure();
    }
  }

  AddUserOpInputOutputSegments(op, attr_vec);

  return success();
}

LogicalResult Importer::AppendCtrlInOperand(const ::oneflow::OperatorConf& op,
                                            std::vector<::mlir::Value>& operand_vec) {
  for (auto ctrl_in_op_name : op.ctrl_in_op_name()) {
    if (op_name2ctrl_result_.find(ctrl_in_op_name) == op_name2ctrl_result_.end()) {
      module_.emitError("IR result not found for ctrl in op: " + ctrl_in_op_name);
      return failure();
    } else {
      auto v = op_name2ctrl_result_.at(ctrl_in_op_name);
      operand_vec.push_back(v);
    }
  }
  return success();
}

LogicalResult Importer::AppendDataInOperand(const std::string& lbn,
                                            std::vector<::mlir::Value>& operand_vec) {
  if (lbn2result_.find(lbn) == lbn2result_.end()) {
    module_.emitError("IR result not found for: " + lbn);
    return failure();
  } else {
    auto v = lbn2result_.at(lbn);
    operand_vec.push_back(v);
    return success();
  }
}

LogicalResult Importer::AddOperandSegmentSizes(int input_lbns_size, int ctrl_in_size,
                                               std::vector<NamedAttribute>& attr_vec) {
  attr_vec.push_back(builder_.getNamedAttr(
      "operand_segment_sizes", builder_.getI32VectorAttr({input_lbns_size, ctrl_in_size})));
  return success();
}

LogicalResult Importer::AddResultSegmentSizes(int output_lbns_size,
                                              std::vector<NamedAttribute>& attr_vec) {
  attr_vec.push_back(builder_.getNamedAttr("result_segment_sizes",
                                           builder_.getI32VectorAttr({output_lbns_size, 1})));
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

ResultRange GetDataOutputResults(Operation* op) {
  if (op->hasAttrOfType<::mlir::DenseIntElementsAttr>("result_segment_sizes")) {
    return getODSResults(op, 0);
  } else {
    return op->getOpResults();
  }
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

LogicalResult Importer::InsertOpResults(Operation* created_op) {
  for (auto data_out : llvm::enumerate(GetDataOutputResults(created_op))) {
    auto output_lbns = created_op->getAttrOfType<ArrayAttr>("output_lbns");
    lbn2result_.insert({output_lbns[data_out.index()].dyn_cast<StringAttr>().getValue().str(),
                        data_out.value().dyn_cast<OpResult>()});
  }
  if (auto ctrl_out = GetCtrlOutputResult(created_op)) {
    op_name2ctrl_result_.insert({created_op->getAttrOfType<StringAttr>("op_name").getValue().str(),
                                 ctrl_out->dyn_cast<OpResult>()});
  }
  return success();
}

LogicalResult Importer::AppendCtrlOutType(llvm::SmallVector<Type, 8>& out_types) {
  out_types.append({RankedTensorType::get({}, builder_.getI1Type())});
  return success();
}

LogicalResult Importer::AddPlacement(const ::oneflow::OperatorConf& op,
                                     std::vector<NamedAttribute>& attr_vec) {
  const ::oneflow::ParallelConf& pc = job_wrapper_.ParallelConf4OpName(op.name());
  std::vector<llvm::StringRef> device_vec = {pc.device_name().begin(), pc.device_name().end()};
  attr_vec.push_back(builder_.getNamedAttr("placement", builder_.getStrArrayAttr(device_vec)));
  return success();
}

LogicalResult Importer::AddOpConf(const ::oneflow::OperatorConf& op,
                                  std::vector<NamedAttribute>& attr_vec) {
  attr_vec.push_back(builder_.getNamedAttr("op_name", builder_.getStringAttr(op.name())));
  if (op.has_device_tag()) {
    attr_vec.push_back(builder_.getNamedAttr("device", builder_.getStringAttr(op.device_tag())));
  }
  attr_vec.push_back(
      builder_.getNamedAttr("scope_symbol_id", builder_.getI64IntegerAttr(op.scope_symbol_id())));
  return success();
}

LogicalResult Importer::ProcessUserOp(const ::oneflow::OperatorConf& op) {
  if (op.has_user_conf() == false) {
    module_.emitError("Not a user op. op name: " + op.name());
    return failure();
  }
  const ::oneflow::UserOpConf& user_conf = op.user_conf();
  const std::string& op_type_name = user_conf.op_type_name();

  std::vector<NamedAttribute> attr_vec;
  if (failed(AddOpConf(op, attr_vec))) { return failure(); }
  if (failed(AddPlacement(op, attr_vec))) { return failure(); }
  attr_vec.push_back(
      builder_.getNamedAttr("op_type_name", builder_.getStringAttr(op.user_conf().op_type_name())));
  std::vector<::mlir::Value> operand_vec;
  if (failed(namedAttributesFromUserOp(op, attr_vec))) { return failure(); }
  for (auto kv : op.user_conf().input()) {
    for (int i = 0; i < kv.second.s_size(); i++) {
      const std::string& lbn = kv.second.s(i);
      if (failed(AppendDataInOperand(lbn, operand_vec))) { return failure(); }
    }
  }
  AppendCtrlInOperand(op, operand_vec);
  ::mlir::ValueRange operands(operand_vec);

  Operation* created_op = nullptr;
  if (op_type_name == "constant") {
    auto t = user_conf.attr().at("is_floating_value").at_bool()
                 ? RankedTensorType::get({}, builder_.getF32Type())
                 : RankedTensorType::get({}, builder_.getI32Type());
    auto out_types = llvm::SmallVector<Type, 8>();
    out_types.append({t});
    AddOperandSegmentSizes(0, op.ctrl_in_op_name_size(), attr_vec);
    ArrayRef<NamedAttribute> named_attributes(attr_vec);
    created_op = builder_.create<oneflow::ConstantOp>(
        FileLineColLoc::get(op.name(), 0, 0, context_), out_types, operands, named_attributes);
  } else {
    auto out_types = llvm::SmallVector<Type, 8>();
    for (auto kv : op.user_conf().output()) {
      for (int i = 0; i < kv.second.s_size(); i++) {
        out_types.append({RankedTensorType::get({}, builder_.getF32Type())});
      }
    }
    AppendCtrlOutType(out_types);
    // OperationState state(unknownLoc, "oneflow." + op_type_name);
    OperationState state(FileLineColLoc::get(op.name(), 0, 0, context_), "oneflow.user");
    for (auto na : attr_vec) {
      if (na.first.str() == "input_lbn_segment_sizes") {
        int data_input_size = 0;
        for (auto segment_size : na.second.dyn_cast<ArrayAttr>()) {
          data_input_size += segment_size.dyn_cast<IntegerAttr>().getInt();
        }
        AddOperandSegmentSizes(data_input_size, op.ctrl_in_op_name_size(), attr_vec);
      }
      if (na.first.str() == "output_lbns") {
        AddResultSegmentSizes(na.second.dyn_cast<ArrayAttr>().size(), attr_vec);
        if (na.second.dyn_cast<ArrayAttr>().size() != out_types.size() - 1) {
          module_->emitError("out_types - 1 != output_lbns, op: " + op.name());
          return failure();
        }
      }
    }
    ArrayRef<NamedAttribute> named_attributes(attr_vec);
    state.addAttributes(named_attributes);
    state.addOperands(operands);
    state.addTypes(out_types);
    created_op = builder_.createOperation(state);
  }

  if (created_op == nullptr) {
    module_->emitError("fail to create " + op.user_conf().op_type_name()
                       + " op, name: " + op.name());
    return failure();
  }
  InsertOpResults(created_op);

  return success();
}  // namespace

LogicalResult Importer::ProcessSystemOp(const ::oneflow::OperatorConf& op) {
  if (op.has_user_conf()) {
    module_.emitError("Not a sys op. op name: " + op.name());
    return failure();
  }
  auto input_bns_lbns = job_wrapper_.InputBns4OpName(op.name());
  auto input_bns = input_bns_lbns.first;
  auto input_lbns = input_bns_lbns.second;
  auto output_lbns = job_wrapper_.OutputLbns4OpName(op.name());
  job_wrapper_.OutputLbns4OpName(op.name());
  std::vector<NamedAttribute> attr_vec;
  if (failed(AddOpConf(op, attr_vec))) { return failure(); }
  if (failed(AddPlacement(op, attr_vec))) { return failure(); }
  attr_vec.push_back(builder_.getNamedAttr(
      "input_bns", builder_.getStrArrayAttr(
                       std::vector<llvm::StringRef>({input_bns.begin(), input_bns.end()}))));
  attr_vec.push_back(builder_.getNamedAttr(
      "output_lbns", builder_.getStrArrayAttr(
                         std::vector<llvm::StringRef>({output_lbns.begin(), output_lbns.end()}))));
  OperationState state(FileLineColLoc::get(op.name(), 0, 0, context_), "oneflow.system");
  attr_vec.push_back(
      builder_.getNamedAttr("op_type_case", builder_.getI32IntegerAttr(op.op_type_case())));
  AddOperandSegmentSizes(static_cast<int>(input_lbns.size()), op.ctrl_in_op_name_size(), attr_vec);
  AddResultSegmentSizes(output_lbns.size(), attr_vec);
  state.addAttributes(attr_vec);
  std::vector<::mlir::Value> operand_vec;
  for (auto input_lbn : input_lbns) {
    if (failed(AppendDataInOperand(input_lbn, operand_vec))) { return failure(); }
  }
  AppendCtrlInOperand(op, operand_vec);
  auto out_types = llvm::SmallVector<Type, 8>();
  for (auto output_lbn : output_lbns) {
    out_types.append({RankedTensorType::get({}, builder_.getF32Type())});
  }
  AppendCtrlOutType(out_types);
  state.addOperands(operand_vec);
  state.addTypes(out_types);
  auto created_op = builder_.createOperation(state);
  InsertOpResults(created_op);

  if (!created_op) {
    module_->emitError("fail to create op, name: " + op.name());
    return failure();
  }
  return success();
}

LogicalResult Importer::ProcessJob() {
  auto func_type = builder_.getFunctionType(llvm::None, llvm::None);
  auto function = mlir::FuncOp::create(unknown_loc_, job_->job_conf().job_name(), func_type);
  auto& entryBlock = *function.addEntryBlock();
  builder_.setInsertionPointToStart(&entryBlock);

  bool is_succeeded = true;
  job_wrapper_.TopoForEachOpConf([&](const ::oneflow::OperatorConf* op_conf) {
    const auto op = *op_conf;
    if (is_succeeded == false) { return; }
    if (op.has_user_conf()) {
      is_succeeded = succeeded(ProcessUserOp(op));
    } else {
      is_succeeded = succeeded(ProcessSystemOp(op));
    }
  });
  if (is_succeeded == false) { return failure(); }

  ReturnOp returnOp;
  if (!entryBlock.empty()) { returnOp = dyn_cast<ReturnOp>(entryBlock.back()); }
  if (!returnOp) { builder_.create<ReturnOp>(unknown_loc_); }
  module_.push_back(function);
  return success();
}

void ConvertCtrlInputs(Operation* op, ::oneflow::OperatorConf& op_conf) {
  if (auto ctrl_ins = GetCtrlIntputOperands(op)) {
    for (auto ctrl_in : ctrl_ins.getValue()) {
      op_conf.add_ctrl_in_op_name(
          ctrl_in.getDefiningOp()->getAttrOfType<StringAttr>("op_name").getValue().str());
    }
  }
}

void ConvertUseropInputs(Operation* op, ::oneflow::UserOpConf* user_conf, std::string& err_str) {
  const std::string op_name = op->getAttrOfType<StringAttr>("op_name").getValue().str();
  int input_idx = 0;
  if (auto keys = op->getAttrOfType<ArrayAttr>("input_lbn_segment_keys")) {
    auto sizes = op->getAttrOfType<ArrayAttr>("input_lbn_segment_sizes");
    if (keys.size() != sizes.size()) {
      err_str = "fail to convert op inputs, input_lbn_segment_keys != "
                "input_lbn_segment_sizes, name: "
                + op_name;
      return;
    };
    // every key
    for (int key_idx = 0; key_idx < keys.size(); key_idx++) {
      auto input_key = keys[key_idx].dyn_cast<StringAttr>().getValue().str();
      auto input_size = sizes[key_idx].dyn_cast<IntegerAttr>().getInt();
      // every input for one key
      for (int i = 0; i < input_size; i++) {
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
          err_str = "fail to cast result, name: " + op_name;
          return;
        }
      }
    }
  } else {
    err_str = "fail to convert op inputs, name: " + op_name;
    return;
  }
}

void ConvertUseropOutputs(Operation* op, ::oneflow::UserOpConf* user_conf, std::string& err_str) {
  int output_key_idx = -1;
  int segment_offset = 0;
  for (auto result_and_idx : llvm::enumerate(GetDataOutputResults(op))) {
    const size_t result_idx = result_and_idx.index();
    if (result_idx == segment_offset) {
      output_key_idx += 1;
      int size = op->getAttrOfType<ArrayAttr>("output_lbn_segment_sizes")[output_key_idx]
                     .dyn_cast<IntegerAttr>()
                     .getInt();
      segment_offset += size;
    }
    std::string output_key = op->getAttrOfType<ArrayAttr>("output_lbn_segment_keys")[output_key_idx]
                                 .dyn_cast<StringAttr>()
                                 .getValue()
                                 .str();
    std::string output_lbn = op->getAttrOfType<ArrayAttr>("output_lbns")[result_idx]
                                 .dyn_cast<StringAttr>()
                                 .getValue()
                                 .str();
    *((*user_conf->mutable_output())[output_key].mutable_s()->Add()) = output_lbn;
  }
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

void Importer::ConvertUseropAttributes(Operation* op, ::oneflow::OperatorConf& op_conf,
                                       std::string& err_str) {
  auto user_conf = op_conf.mutable_user_conf();
  const std::string op_name = op->getAttrOfType<StringAttr>("op_name").getValue().str();
  for (auto id_attr : op->getAttrDictionary()) {
    auto id = id_attr.first;

    if (id.strref().equals("placement") || id.strref().contains("input_lbn_segment_keys")
        || id.strref().contains("input_lbn_segment_sizes") || id.strref().contains("output_lbns")
        || id.strref().contains("output_lbn_segment_keys")
        || id.strref().contains("output_lbn_segment_sizes")
        || id.strref().contains("operand_segment_sizes")
        || id.strref().contains("result_segment_sizes")) {
      continue;
    }

    // convert op conf attributes
    if (id.strref().equals("op_name")) {
      op_conf.set_name(op_name);
      continue;
    }
    if (id.strref().equals("op_type_name")) {
      user_conf->set_op_type_name(op->getAttrOfType<StringAttr>("op_type_name").getValue().str());
      continue;
    }
    if (id.strref().equals("device")) {
      op_conf.set_device_tag(op->getAttrOfType<StringAttr>("device").getValue().str());
      continue;
    }
    if (id.strref().equals("scope_symbol_id")) {
      op_conf.set_scope_symbol_id(op->getAttrOfType<IntegerAttr>("scope_symbol_id").getInt());
      continue;
    }  // convert op conf attributes

    // convert user conf attributes
    auto attr_name = id.str();
    Attribute attr = id_attr.second;
    auto user_attr = ::oneflow::AttrValue();
    auto op_type_name = op->getAttrOfType<StringAttr>("op_type_name").getValue().str();
    ::oneflow::AttrType attr_type = job_wrapper_.QueryAttrType(op_type_name, attr_name);
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
      for (auto int_v : attr.dyn_cast<DenseIntElementsAttr>().getIntValues()) {
        assert(int_v.isSignedIntN(64));
        user_attr.mutable_at_shape()->add_dim(int_v.getSExtValue());
      }
    } else if (attr_type == ::oneflow::kAtDataType) {
      ::oneflow::DataType dt;
      if (succeeded(ConvertDT(attr, dt))) {
        user_attr.set_at_data_type(dt);
      } else {
        err_str = "fail to convert op attr to data type, key: " + id.str();
        return;
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
        ::oneflow::DataType dt;
        if (succeeded(ConvertDT(v, dt))) {
          user_attr.mutable_at_list_data_type()->add_val(dt);
        } else {
          err_str = "fail to convert op attr to data type, key: " + id.str();
          return;
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
      for (auto s : attr.dyn_cast<ArrayAttr>().getValue()) {
        user_attr.mutable_at_list_string()->add_val(s.dyn_cast<StringAttr>().getValue().str());
      }
    } else {
      err_str = "fail to convert op attr of name: " + attr_name;
      return;
    }  // convert user conf attributes

    (*user_conf->mutable_attr())[id.str()] = user_attr;
  }
}

LogicalResult Importer::TryToUpdateJob() {
  std::string err_str = "";
  auto new_job = ::oneflow::Job();
  new_job.CopyFrom(*job_);
  new_job.clear_net();
  auto convertOps = [&](Operation* op) {
    if (llvm::dyn_cast<oneflow::UserOp>(op) || op->hasAttr("op_type_name")) {
      ::oneflow::OperatorConf op_conf;
      const std::string op_name = op->getAttrOfType<StringAttr>("op_name").getValue().str();
      auto user_conf = op_conf.mutable_user_conf();
      ConvertUseropInputs(op, user_conf, err_str);
      ConvertUseropOutputs(op, user_conf, err_str);
      ConvertUseropAttributes(op, op_conf, err_str);
      ConvertCtrlInputs(op, op_conf);
      *(new_job.mutable_net()->add_op()) = op_conf;
    } else if (llvm::dyn_cast<oneflow::SystemOp>(op)) {
      auto op_name = op->getAttrOfType<StringAttr>("op_name").getValue().str();
      ::oneflow::OperatorConf op_conf = job_wrapper_.OpConf4OpName(op_name);
      for (auto ibn : llvm::enumerate(op->getAttrOfType<ArrayAttr>("input_bns"))) {
        auto result = GetDataInputOperands(op)[ibn.index()].dyn_cast<OpResult>();
        std::string new_val =
            result.getDefiningOp()
                ->getAttrOfType<ArrayAttr>("output_lbns")[result.getResultNumber()]
                .dyn_cast<StringAttr>()
                .getValue()
                .str();
        job_wrapper_.ReplaceInputLbnInOpCustomizedConf(
            &op_conf, ibn.value().dyn_cast<StringAttr>().getValue().str(), new_val);
      }
      ConvertCtrlInputs(op, op_conf);
      *(new_job.mutable_net()->add_op()) = op_conf;
    } else if (llvm::dyn_cast<ModuleTerminatorOp>(op)) {
      // Do nothing
    } else if (llvm::dyn_cast<ReturnOp>(op)) {
      // Do nothing
    } else if (llvm::dyn_cast<FuncOp>(op)) {
      // Do nothing
    } else {
      err_str = "failed to convert MLIR op: " + op->getName().getStringRef().str();
      op->dump();
      return;
    } /* convert op conf */
  };
  module_.getBodyRegion().walk(convertOps);
  if (err_str.empty()) {
    job_wrapper_.UpdateJob(&new_job);
    return success();
  } else {
    module_->emitError(err_str);
    return failure();
  }
}  // namespace

LogicalResult applyRoundTripPatterns(MLIRContext* context, OwningModuleRef& module, bool debug) {
  if (debug) {
    std::cout << "import:\n";
    module->dump();
  }

  mlir::PassManager pm(context);
  pm.addNestedPass<mlir::FuncOp>(::mlir::createCanonicalizerPass());
  if (mlir::failed(pm.run(*module))) {
    module->emitError("Failed to run canonicalizer pass");
    return failure();
  }

  if (debug) {
    std::cout << "optimized:\n";
    module->dump();
  }
  return success();
}

// Move this into another cpp which will be another target
OwningModuleRef translateOneFlowJobToModule(llvm::StringRef str, MLIRContext* context) {
  std::string cpp_str = str.str();
  ::oneflow::Job job;
  google::protobuf::TextFormat::ParseFromString(cpp_str, &job);
  context->loadDialect<oneflow::OneFlowDialect>();
  context->loadDialect<StandardOpsDialect>();
  // Reimplement the logic after this function is moved to a independent target
  OwningModuleRef module(
      ModuleOp::create(FileLineColLoc::get("", /*line=*/0, /*column=*/0, context)));
  return module;
}
}  // namespace

void RoundTripOneFlowJob(
    RoundTripOneFlowJobWrapperInterface& job_wrapper,
    std::function<bool(::oneflow::Job* job, std::string& reason)> is_legit_job) {
  const ::oneflow::Job* job = job_wrapper.job();
  mlir::MLIRContext context;
  // Load our Dialect in this MLIR Context.
  context.getOrLoadDialect<oneflow::OneFlowDialect>();
  context.loadDialect<StandardOpsDialect>();
  OwningModuleRef module(
      ModuleOp::create(FileLineColLoc::get("", /*line=*/0, /*column=*/0, &context)));
  Importer imp(job_wrapper, &context, module.get());
  // TODO: Add flag in job desc
  const bool is_strict = true;
  if (succeeded(imp.ProcessJob())) {
    if (failed(applyRoundTripPatterns(&context, module,
                                      std::getenv("ONEFLOW_DEBUG_MODE") != nullptr))) {
      exit(EXIT_FAILURE);
    }
    std::string mlir;
    llvm::raw_string_ostream os(mlir);
    module->print(os);
    job_wrapper.DumpMLIR("RoundTripOneFlowJobuilder_." + job->job_conf().job_name() + ".mlir",
                         mlir);

    if (failed(imp.TryToUpdateJob())) {
      std::cerr << "fail to update job with IR, job will stay intact, job_name: "
                << job->job_conf().job_name() << "\n";
      if (is_strict) { exit(EXIT_FAILURE); }
    }
  } else {
    std::cerr << "fail to convert job to IR, job_name: " << job->job_conf().job_name() << "\n";
    if (is_strict) { exit(EXIT_FAILURE); }
  }
}

void registerFromOneFlowJobTranslation() {
  TranslateToMLIRRegistration fromOneFlowJob("import-oneflow-job",
                                             [](llvm::StringRef str, MLIRContext* context) {
                                               return translateOneFlowJobToModule(str, context);
                                             });
}

}  // namespace mlir
