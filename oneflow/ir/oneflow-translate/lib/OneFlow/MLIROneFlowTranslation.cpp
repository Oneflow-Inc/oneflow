#include "OneFlow/OneFlowOps.h"
#include "llvm-c/Core.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Translation.h"
#include "mlir/IR/Builders.h"

#include "OneFlow/OneFlowDialect.h"

#include <google/protobuf/text_format.h>
#include "oneflow/core/common/data_type.pb.h"
#include "oneflow/core/framework/user_op_attr.pb.h"
#include "oneflow/core/framework/user_op_conf.pb.h"
#include "oneflow/core/job/job.pb.h"
#include "oneflow/core/operator/op_conf.pb.h"
#include <cstddef>
#include <cstdint>
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

Attribute createEmptyDictionaryAttr(Builder &builder) { return builder.getDictionaryAttr({}); }
::mlir::ValueRange putInVariadic(Builder &builder, Value v) {
  ::mlir::ValueRange operands({v});
  return operands;
}

using PbMessage = google::protobuf::Message;

class Importer {
 public:
  Importer(RoundTripOneFlowJobWrapperInterface &job_wrapper, MLIRContext *context, ModuleOp module)
      : b(context),
        context(context),
        module(module),
        unknownLoc(FileLineColLoc::get("imported-protobuf", 0, 0, context)),
        job(job_wrapper.job()),
        job_wrapper(job_wrapper) {}

  LogicalResult namedAttributesFromUserOp(const ::oneflow::OperatorConf &op,
                                          std::vector<NamedAttribute> &attr_vec);
  LogicalResult operandsFromUserOp(const ::oneflow::OperatorConf &op,
                                   std::vector<::mlir::Value> &operand_vec);
  LogicalResult AddInputOutputSegments(const ::oneflow::OperatorConf &op,
                                       std::vector<NamedAttribute> &attr_vec);
  LogicalResult InsertOpResults(Operation *);
  LogicalResult processUserOp(const ::oneflow::OperatorConf &op);
  LogicalResult processSystemOp(const ::oneflow::OperatorConf &op);
  LogicalResult processJob();
  LogicalResult tryToUpdateJob();

  void ConvertUseropAttributes(Operation *op, ::oneflow::OperatorConf &op_conf,
                               std::string &err_str);

  IntegerAttr getSI64IntegerAttr(int64_t value) {
    return IntegerAttr::get(b.getIntegerType(64, /*isSigned=*/true),
                            APInt(64, value, /*isSigned=*/true));
  }
  ArrayAttr getSI32ArrayAttr(ArrayRef<int32_t> values) {
    auto attrs = llvm::to_vector<8>(llvm::map_range(
        values, [this](int32_t v) -> Attribute { return b.getSI32IntegerAttr(v); }));
    return b.getArrayAttr(attrs);
  }
  ArrayAttr getSI64ArrayAttr(ArrayRef<int64_t> values) {
    auto attrs = llvm::to_vector<8>(
        llvm::map_range(values, [this](int64_t v) -> Attribute { return getSI64IntegerAttr(v); }));
    return b.getArrayAttr(attrs);
  }

 private:
  /// The current builder, pointing at where the next Instruction should be
  /// generated.
  OpBuilder b;
  /// The current context.
  MLIRContext *context;
  /// The current module being created.
  ModuleOp module;
  /// Cached FileLineColLoc::get("imported-protobuf", 0, 0).
  Location unknownLoc;
  std::unordered_map<std::string, mlir::Value> lbn2result_;
  const ::oneflow::Job *job;
  RoundTripOneFlowJobWrapperInterface &job_wrapper;
};

// TODO: add trait for this
LogicalResult Importer::AddInputOutputSegments(const ::oneflow::OperatorConf &op,
                                               std::vector<NamedAttribute> &attr_vec) {
  std::vector<llvm::StringRef> input_lbn_segment_keys;
  std::vector<int> input_lbn_segment_sizes;
  for (auto input : op.user_conf().input()) {
    input_lbn_segment_keys.push_back(input.first);
    input_lbn_segment_sizes.push_back(input.second.s_size());
  }
  attr_vec.push_back(
      b.getNamedAttr("input_lbn_segment_keys", b.getStrArrayAttr(input_lbn_segment_keys)));
  attr_vec.push_back(
      b.getNamedAttr("input_lbn_segment_sizes", b.getI32ArrayAttr(input_lbn_segment_sizes)));

  std::vector<llvm::StringRef> output_lbns;
  std::vector<llvm::StringRef> output_lbn_segment_keys;
  std::vector<int> output_lbn_segment_sizes;
  int data_output_size = 0;
  for (auto output : op.user_conf().output()) {
    output_lbns.insert(output_lbns.end(), output.second.s().begin(), output.second.s().end());
    output_lbn_segment_keys.push_back(output.first);
    output_lbn_segment_sizes.push_back(output.second.s_size());
    data_output_size += output.second.s_size();
    // TODO: merge this method with output types create
  }
  attr_vec.push_back(b.getNamedAttr("output_lbns", b.getStrArrayAttr(output_lbns)));
  attr_vec.push_back(
      b.getNamedAttr("output_lbn_segment_keys", b.getStrArrayAttr(output_lbn_segment_keys)));
  attr_vec.push_back(
      b.getNamedAttr("output_lbn_segment_sizes", b.getI32ArrayAttr(output_lbn_segment_sizes)));
  attr_vec.push_back(
      b.getNamedAttr("result_segment_sizes", b.getI32VectorAttr({data_output_size, 1})));
  return success();
}

LogicalResult Importer::namedAttributesFromUserOp(const ::oneflow::OperatorConf &op,
                                                  std::vector<NamedAttribute> &attr_vec) {
  if (op.has_user_conf() == false) {
    module.emitError("Not a user op. op name: " + op.name());
    return failure();
  }
  for (const google::protobuf::MapPair<class std::basic_string<char>, ::oneflow::AttrValue> &attr :
       op.user_conf().attr()) {
    const std::string &name = attr.first;
    const ::oneflow::AttrValue &value = attr.second;
    if (value.has_at_int32()) {
      std::pair<mlir::Identifier, mlir::Attribute> kv =
          b.getNamedAttr(name, b.getSI32IntegerAttr(value.at_int32()));
      attr_vec.emplace_back(kv);
    } else if (value.has_at_int64()) {
      std::pair<mlir::Identifier, mlir::Attribute> kv =
          b.getNamedAttr(name, getSI64IntegerAttr(value.at_int64()));
      attr_vec.emplace_back(kv);
    }
#define DEFINE_ONE_ELIF(at_key, get_attr)                 \
  else if (value.has_##at_key()) {                        \
    std::pair<mlir::Identifier, mlir::Attribute> kv =     \
        b.getNamedAttr(name, b.get_attr(value.at_key())); \
    attr_vec.emplace_back(kv);                            \
  }
    DEFINE_ONE_ELIF(at_bool, getBoolAttr)
    DEFINE_ONE_ELIF(at_float, getF32FloatAttr)
    DEFINE_ONE_ELIF(at_double, getF64FloatAttr)
    DEFINE_ONE_ELIF(at_string, getStringAttr)
#undef DEFINE_ONE_ELIF
    else if (value.has_at_shape()) {
      ArrayRef<int64_t> values = {value.at_shape().dim().begin(), value.at_shape().dim().end()};
      RankedTensorType tt =
          RankedTensorType::get({static_cast<int64_t>(values.size())}, b.getIntegerType(64, true));
      ;
      attr_vec.emplace_back(b.getNamedAttr(name, DenseIntElementsAttr::get(tt, values)));
    }
#define DEFINE_ONE_ELIF(at_key, get_attr, field)                                         \
  else if (value.has_##at_key()) {                                                       \
    std::pair<mlir::Identifier, mlir::Attribute> kv = b.getNamedAttr(                    \
        name, get_attr({value.at_key().field().begin(), value.at_key().field().end()})); \
    attr_vec.emplace_back(kv);                                                           \
  }
    DEFINE_ONE_ELIF(at_list_int32, getSI32ArrayAttr, val)
    DEFINE_ONE_ELIF(at_list_int64, getSI64ArrayAttr, val)
    DEFINE_ONE_ELIF(at_list_float, b.getF32ArrayAttr, val)
#undef DEFINE_ONE_ELIF
    else if (value.has_at_list_string()) {
      std::vector<llvm::StringRef> r_vec = {value.at_list_string().val().begin(),
                                            value.at_list_string().val().end()};
      std::pair<mlir::Identifier, mlir::Attribute> kv =
          b.getNamedAttr(name, b.getStrArrayAttr(r_vec));
      attr_vec.emplace_back(kv);
    }
    else if (value.has_at_data_type()) {
      std::string stringified = "";
      switch (value.at_data_type()) {
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
        default: module.emitError("fail to convert op attr, key: " + name); break;
      }
      std::pair<mlir::Identifier, mlir::Attribute> kv =
          b.getNamedAttr(name, b.getStringAttr(stringified));
      attr_vec.emplace_back(kv);
    }
    else {
      module.emitError("can't handle user op attr: " + name + ", op name: " + op.name()
                       + ", op type name: " + op.user_conf().op_type_name());
      return failure();
    }
  }

  AddInputOutputSegments(op, attr_vec);

  return success();
}

LogicalResult Importer::operandsFromUserOp(const ::oneflow::OperatorConf &op,
                                           std::vector<Value> &operand_vec) {
  if (op.has_user_conf() == false) {
    module.emitError("Not a user op. op name: " + op.name());
    return failure();
  }
  for (auto kv : op.user_conf().input()) {
    for (int i = 0; i < kv.second.s_size(); i++) {
      const std::string &lbn = kv.second.s(i);
      if (lbn2result_.find(lbn) == lbn2result_.end()) {
        module.emitError("IR result not found for: " + lbn);
        return failure();
      } else {
        auto v = lbn2result_.at(lbn);
        operand_vec.push_back(v);
      }
    }
  }
  // TODO: add result lookup for ctrl in
  return success();
}

LogicalResult Importer::InsertOpResults(Operation *created_op) {
  for (auto output_lbn : llvm::enumerate(created_op->getAttrOfType<ArrayAttr>("output_lbns"))) {
    lbn2result_.insert({output_lbn.value().dyn_cast<StringAttr>().getValue().str(),
                        created_op->getResult(output_lbn.index())});
  }
  return success();
}

LogicalResult Importer::processUserOp(const ::oneflow::OperatorConf &op) {
  if (op.has_user_conf() == false) {
    module.emitError("Not a user op. op name: " + op.name());
    return failure();
  }
  const ::oneflow::UserOpConf &user_conf = op.user_conf();
  const std::string &op_type_name = user_conf.op_type_name();
  const std::string &op_name = op.name();
  const ::oneflow::ParallelConf &pc = job_wrapper.ParallelConf4OpName(op_name);
  std::vector<llvm::StringRef> device_vec = {pc.device_name().begin(), pc.device_name().end()};
  std::vector<NamedAttribute> attr_vec;
  attr_vec.push_back(b.getNamedAttr("op_name", b.getStringAttr(op.name())));
  if (op.has_trainable()) {
    attr_vec.push_back(b.getNamedAttr("trainable", b.getBoolAttr(op.trainable())));
  }
  if (op.has_device_tag()) {
    attr_vec.push_back(b.getNamedAttr("device", b.getStringAttr(op.device_tag())));
  }
  attr_vec.push_back(b.getNamedAttr("placement", b.getStrArrayAttr(device_vec)));
  attr_vec.push_back(b.getNamedAttr("scope_symbol_id", b.getI64IntegerAttr(op.scope_symbol_id())));
  attr_vec.push_back(
      b.getNamedAttr("op_type_name", b.getStringAttr(op.user_conf().op_type_name())));
  std::vector<::mlir::Value> operand_vec;
  if (failed(namedAttributesFromUserOp(op, attr_vec))) { return failure(); }
  ArrayRef<NamedAttribute> named_attributes(attr_vec);
  if (failed(operandsFromUserOp(op, operand_vec))) { return failure(); }
  ::mlir::ValueRange operands(operand_vec);

  Operation *created_op = nullptr;

  if (op_type_name == "constant") {
    auto t = user_conf.attr().at("is_floating_value").at_bool()
                 ? RankedTensorType::get({}, b.getF32Type())
                 : RankedTensorType::get({}, b.getI32Type());
    created_op = b.create<oneflow::ConstantOp>(unknownLoc, t, operands, named_attributes);
  } else {
    auto out_types = llvm::SmallVector<Type, 8>();
    for (auto kv : op.user_conf().output()) {
      for (int i = 0; i < kv.second.s_size(); i++) {
        out_types.append({RankedTensorType::get({}, b.getF32Type())});
      }
    }
    // add one result for ctrl out
    out_types.append({RankedTensorType::get({}, b.getI1Type())});
    OperationState state(unknownLoc, "oneflow." + op_type_name);
    state.addAttributes(named_attributes);
    state.addOperands(operands);
    state.addTypes(out_types);
    created_op = b.createOperation(state);
  }

  if (created_op == nullptr) {
    module->emitError("fail to create " + op.user_conf().op_type_name()
                      + " op, name: " + op.name());
    return failure();
  }
  InsertOpResults(created_op);

  return success();
}  // namespace

LogicalResult Importer::processSystemOp(const ::oneflow::OperatorConf &op) {
  if (op.has_user_conf()) {
    module.emitError("Not a sys op. op name: " + op.name());
    return failure();
  }
  auto input_bns_lbns = job_wrapper.InputBns4OpName(op.name());
  auto input_bns = input_bns_lbns.first;
  auto input_lbns = input_bns_lbns.second;
  auto output_lbns = job_wrapper.OutputLbns4OpName(op.name());
  job_wrapper.OutputLbns4OpName(op.name());
  std::vector<NamedAttribute> attr_vec;
  attr_vec.push_back(b.getNamedAttr("input_bns", b.getStrArrayAttr(std::vector<llvm::StringRef>(
                                                     {input_bns.begin(), input_bns.end()}))));
  attr_vec.push_back(b.getNamedAttr("output_lbns", b.getStrArrayAttr(std::vector<llvm::StringRef>(
                                                       {output_lbns.begin(), output_lbns.end()}))));
  OperationState state(unknownLoc, "oneflow.system");
  attr_vec.push_back(b.getNamedAttr("op_type_case", b.getI32IntegerAttr(op.op_type_case())));
  attr_vec.push_back(b.getNamedAttr("op_name", b.getStringAttr(op.name())));
  if (op.ctrl_in_op_name_size() > 0) {
    // TODO: get ctrl result from a {op_name => ctrl_result} map
  }
  state.addAttributes(attr_vec);
  std::vector<::mlir::Value> operand_vec;
  for (auto input_lbn : input_lbns) {
    if (lbn2result_.find(input_lbn) == lbn2result_.end()) {
      module.emitError("IR result not found for: " + input_lbn);
      return failure();
    } else {
      auto v = lbn2result_.at(input_lbn);
      operand_vec.push_back(v);
    }
  }
  auto out_types = llvm::SmallVector<Type, 8>();
  for (auto output_lbn : output_lbns) {
    out_types.append({RankedTensorType::get({}, b.getF32Type())});
  }
  state.addOperands(operand_vec);
  state.addTypes(out_types);
  auto created_op = b.createOperation(state);
  InsertOpResults(created_op);

  if (!created_op) {
    module->emitError("fail to create op, name: " + op.name());
    return failure();
  }
  return success();
}

LogicalResult Importer::processJob() {
  auto func_type = b.getFunctionType(llvm::None, llvm::None);
  auto function = mlir::FuncOp::create(unknownLoc, job->job_conf().job_name(), func_type);
  auto &entryBlock = *function.addEntryBlock();
  b.setInsertionPointToStart(&entryBlock);

  for (int64_t i = 0; i < job->net().op_size(); i++) {
    const auto op = job->net().op(i);
    if (op.has_user_conf()) {
      if (failed(processUserOp(op))) { return failure(); }
    } else {
      if (failed(processSystemOp(op))) { return failure(); }
    }
  }
  ReturnOp returnOp;
  if (!entryBlock.empty()) { returnOp = dyn_cast<ReturnOp>(entryBlock.back()); }
  if (!returnOp) { b.create<ReturnOp>(unknownLoc); }
  module.push_back(function);
  return success();
}

void ConvertUseropInputs(Operation *op, ::oneflow::UserOpConf *user_conf, std::string &err_str) {
  const std::string op_name = op->getAttrOfType<StringAttr>("op_name").getValue().str();
  int input_idx = 0;
  if (auto keys = op->getAttrOfType<ArrayAttr>("input_lbn_segment_keys")) {
    auto sizes = op->getAttrOfType<ArrayAttr>("input_lbn_segment_sizes");
    if (keys.size() == sizes.size()) {
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
            *((*user_conf->mutable_input())[input_key].mutable_s()->Add()) =
                output_lbn_in_source_op;
            input_idx += 1;
          } else {
            err_str = "fail to cast result, name: " + op_name;
            return;
          }
        }
      }
    } else {
      err_str =
          "fail to convert op inputs, input_lbn_segment_keys != input_lbn_segment_sizes, name: "
          + op_name;
      return;
    }
  } else {
    err_str = "fail to convert op inputs, name: " + op_name;
    return;
  }
}

void ConvertUseropOutputs(Operation *op, ::oneflow::UserOpConf *user_conf, std::string &err_str) {
  int output_key_idx = -1;
  int segment_offset = 0;
  auto result_segment_sizes = op->getAttrOfType<ElementsAttr>("result_segment_sizes");
  const int64_t output_size = result_segment_sizes.getValue(0).dyn_cast<IntegerAttr>().getInt();
  auto data_out_results = op->getOpResults().take_front(output_size);
  for (auto result_and_idx : llvm::enumerate(data_out_results)) {
    // TODO: reject if it is a ctrl edge
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

void Importer::ConvertUseropAttributes(Operation *op, ::oneflow::OperatorConf &op_conf,
                                       std::string &err_str) {
  auto user_conf = op_conf.mutable_user_conf();
  const std::string op_name = op->getAttrOfType<StringAttr>("op_name").getValue().str();
  for (auto id_attr : op->getAttrDictionary()) {
    auto id = id_attr.first;
    // convert op conf attributes
    if (id.strref().equals("op_name")) {
      op_conf.set_name(op_name);
      continue;
    }
    if (id.strref().equals("op_type_name")) {
      user_conf->set_op_type_name(op->getAttrOfType<StringAttr>("op_type_name").getValue().str());
      continue;
    }
    if (id.strref().equals("trainable")) {
      op_conf.set_trainable(op->getAttrOfType<BoolAttr>("trainable").getValue());
      continue;
    }
    if (id.strref().equals("device")) {
      op_conf.set_device_tag(op->getAttrOfType<StringAttr>("device").getValue().str());
      continue;
    }
    if (id.strref().equals("scope_symbol_id")) {
      op_conf.set_scope_symbol_id(op->getAttrOfType<IntegerAttr>("scope_symbol_id").getInt());
      continue;
    }
    if (id.strref().equals("placement") || id.strref().contains("input_lbn_segment_keys")
        || id.strref().contains("input_lbn_segment_sizes") || id.strref().contains("output_lbns")
        || id.strref().contains("output_lbn_segment_keys")
        || id.strref().contains("output_lbn_segment_sizes")) {
      continue;
    }  // convert op conf attributes

    // convert user conf attributes
    auto attr = id_attr.second;
    auto user_attr = ::oneflow::AttrValue();
    if (auto ref = attr.dyn_cast<BoolAttr>()) /* handle bool before int because it is i1*/ {
      user_attr.set_at_bool(ref.getValue());
    } else if (auto ref = attr.dyn_cast<IntegerAttr>()) {
      if (ref.getType().isSignedInteger(32)) {
        user_attr.set_at_int32(ref.getSInt());
      } else if (ref.getType().isSignedInteger(64)) {
        user_attr.set_at_int64(ref.getSInt());
      } else {
        err_str =
            "fail to convert op attr to int32 or int64, key: " + id.str() + ", op name: " + op_name;
        return;
      }
    } else if (auto ref = attr.dyn_cast<FloatAttr>()) {
      if (ref.getType() == b.getF32Type()) {
        user_attr.set_at_float(ref.getValue().convertToFloat());
      } else if (ref.getType() == b.getF64Type()) {
        user_attr.set_at_double(ref.getValue().convertToDouble());
      } else {
        err_str =
            "fail to convert op attr float or double, key: " + id.str() + ", op name: " + op_name;
        return;
      }
    } else if (auto ref = attr.dyn_cast<StringAttr>()) {
      if (ref.getValue().startswith("DT_")) {
        auto dt = oneflow::symbolizeEnum<oneflow::DataType>(ref.getValue());
        if (dt.hasValue()) {
          switch (dt.getValue()) {
            case oneflow::DataType::DT_InvalidDataType:
              user_attr.set_at_data_type(::oneflow::DataType::kInvalidDataType);
              break;
#define DEFINE_ONE_CASE(datatype)                                 \
  case oneflow::DataType::DT_##datatype:                          \
    user_attr.set_at_data_type(::oneflow::DataType::k##datatype); \
    break;
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
            default: err_str = "fail to convert op attr to data type, key: " + id.str(); return;
          }
        }
      } else {
        user_attr.set_at_string(ref.getValue().str());
      }
    } else if (auto ref = attr.dyn_cast<DenseIntElementsAttr>()) {
      for (auto int_v : ref.getIntValues()) {
        user_attr.mutable_at_shape()->add_dim(*int_v.getRawData());
      }
    } else if (auto ref = attr.dyn_cast<ArrayAttr>()) {
      for (auto v : ref.getValue()) {
        if (auto elem = v.dyn_cast<IntegerAttr>()) {
          if (elem.getType().isSignedInteger(32)) {
            user_attr.mutable_at_list_int32()->add_val(elem.getSInt());
          } else if (elem.getType().isSignedInteger(64)) {
            user_attr.mutable_at_list_int64()->add_val(elem.getSInt());
          } else {
            err_str =
                "fail to convert op attr to int list, key: " + id.str() + ", op name: " + op_name;
            op->dump();
            return;
          }
        } else if (auto elem = v.dyn_cast<FloatAttr>()) {
          if (elem.getType() == b.getF32Type()) {
            user_attr.mutable_at_list_float()->add_val(elem.getValue().convertToFloat());
          } else {
            err_str = "fail to convert op attr to float list, key: " + id.str();
            return;
          }
        } else {
          err_str = "fail to convert op attr to list, key: " + id.str();
          return;
        }
      }
    } else {
      err_str = "fail to convert op attr, key: " + id.str();
      return;
    }  // convert user conf attributes

    (*user_conf->mutable_attr())[id.str()] = user_attr;
  }
}

LogicalResult Importer::tryToUpdateJob() {
  std::string err_str = "";
  auto new_job = ::oneflow::Job();
  new_job.CopyFrom(*job);
  new_job.clear_net();
  auto convertOps = [&](Operation *op) {
    if (/* user op */ op->hasAttr("op_type_name")) {
      ::oneflow::OperatorConf op_conf;
      const std::string op_name = op->getAttrOfType<StringAttr>("op_name").getValue().str();
      auto user_conf = op_conf.mutable_user_conf();
      ConvertUseropInputs(op, user_conf, err_str);
      ConvertUseropOutputs(op, user_conf, err_str);
      ConvertUseropAttributes(op, op_conf, err_str);
      *(new_job.mutable_net()->add_op()) = op_conf;
    } else if (/* system op */ auto sys_op = llvm::dyn_cast<oneflow::SystemOp>(op)) {
      auto op_name = op->getAttrOfType<StringAttr>("op_name").getValue().str();
      *(new_job.mutable_net()->add_op()) = job_wrapper.OpConf4OpName(op_name);
    } else {
      // TODO: check if is module_terminator
    } /* convert op conf */
  };
  module.getBodyRegion().walk(convertOps);
  if (err_str.empty()) {
    job_wrapper.UpdateJob(&new_job);
    return success();
  } else {
    module->emitError(err_str);
    return failure();
  }
}  // namespace

void applyRoundTripPatterns(MLIRContext *context, OwningModuleRef &module, bool debug) {
  if (debug) {
    std::cout << "import:\n";
    module->dump();
  }

  OwningRewritePatternList import_patterns;
  auto applied = applyPatternsAndFoldGreedily(module.get(), std::move(import_patterns));
  if (failed(applied)) { module->emitError("Failed to rewrite user ops"); }
  if (debug) {
    std::cout << "optimized:\n";
    module->dump();
  }

  OwningRewritePatternList export_patterns;
  if (failed(applyPatternsAndFoldGreedily(module.get(), std::move(export_patterns)))) {
    module->emitError("Failed to export user ops");
  }

  if (debug) {
    std::cout << "to export:\n";
    module->dump();
  }
}

// Move this into another cpp which will be another target
OwningModuleRef translateOneFlowJobToModule(llvm::StringRef str, MLIRContext *context) {
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
    RoundTripOneFlowJobWrapperInterface &job_wrapper,
    std::function<bool(::oneflow::Job *job, std::string &reason)> is_legit_job) {
  const ::oneflow::Job *job = job_wrapper.job();
  mlir::MLIRContext context;
  // Load our Dialect in this MLIR Context.
  context.getOrLoadDialect<oneflow::OneFlowDialect>();
  context.loadDialect<StandardOpsDialect>();
  OwningModuleRef module(
      ModuleOp::create(FileLineColLoc::get("", /*line=*/0, /*column=*/0, &context)));
  Importer imp(job_wrapper, &context, module.get());
  if (succeeded(imp.processJob())) {
    applyRoundTripPatterns(&context, module, std::getenv("ONEFLOW_DEBUG_MODE") != nullptr);
    if (failed(imp.tryToUpdateJob())) {
      std::cerr << "fail to update job with IR, job will stay intact, job_name: "
                << job->job_conf().job_name() << "\n";
    }
    std::string mlir;
    llvm::raw_string_ostream os(mlir);
    module->print(os);
    job_wrapper.DumpMLIR("RoundTripOneFlowJob." + job->job_conf().job_name() + ".mlir", mlir);
  } else {
    std::cerr << "fail to convert job to IR, job_name: " << job->job_conf().job_name() << "\n";
  }
}

void registerFromOneFlowJobTranslation() {
  TranslateToMLIRRegistration fromOneFlowJob("import-oneflow-job",
                                             [](llvm::StringRef str, MLIRContext *context) {
                                               return translateOneFlowJobToModule(str, context);
                                             });
}

}  // namespace mlir
