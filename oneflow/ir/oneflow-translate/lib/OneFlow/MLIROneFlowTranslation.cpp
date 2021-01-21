#include "OneFlow/OneFlowOps.h"
#include "llvm-c/Core.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Translation.h"
#include "mlir/IR/Builders.h"

#include "OneFlow/OneFlowDialect.h"

#include <google/protobuf/text_format.h>
#include "oneflow/core/framework/user_op_conf.pb.h"
#include "oneflow/core/job/job.pb.h"
#include "oneflow/core/operator/op_conf.pb.h"
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
  LogicalResult processUserOp(const ::oneflow::OperatorConf &op);
  LogicalResult processSystemOp(const ::oneflow::OperatorConf &op);
  LogicalResult processJob();
  LogicalResult tryToUpdateJob();

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
  std::unordered_map<std::string, mlir::Value> lbn2result;
  const ::oneflow::Job *job;
  RoundTripOneFlowJobWrapperInterface &job_wrapper;
};

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
          b.getNamedAttr(name, b.getI32IntegerAttr(value.at_int32()));
      attr_vec.emplace_back(kv);
    }
#define DEFINE_ONE_ELIF(at_key, get_attr)                 \
  else if (value.has_##at_key()) {                        \
    std::pair<mlir::Identifier, mlir::Attribute> kv =     \
        b.getNamedAttr(name, b.get_attr(value.at_key())); \
    attr_vec.emplace_back(kv);                            \
  }
    DEFINE_ONE_ELIF(at_int64, getI64IntegerAttr)
    DEFINE_ONE_ELIF(at_bool, getBoolAttr)
    DEFINE_ONE_ELIF(at_float, getF32FloatAttr)
    DEFINE_ONE_ELIF(at_double, getF64FloatAttr)
    DEFINE_ONE_ELIF(at_string, getStringAttr)
#undef DEFINE_ONE_ELIF
#define DEFINE_ONE_ELIF(at_key, get_attr, field)                                           \
  else if (value.has_##at_key()) {                                                         \
    std::pair<mlir::Identifier, mlir::Attribute> kv = b.getNamedAttr(                      \
        name, b.get_attr({value.at_key().field().begin(), value.at_key().field().end()})); \
    attr_vec.emplace_back(kv);                                                             \
  }
    // TODO: Define a shape attribute type backed by i64 array storage
    DEFINE_ONE_ELIF(at_shape, getI64ArrayAttr, dim)
    DEFINE_ONE_ELIF(at_list_int32, getI32ArrayAttr, val)
    DEFINE_ONE_ELIF(at_list_int64, getI64ArrayAttr, val)
    DEFINE_ONE_ELIF(at_list_float, getF32ArrayAttr, val)
#undef DEFINE_ONE_ELIF
    else if (value.has_at_list_string()) {
      std::vector<llvm::StringRef> r_vec = {value.at_list_string().val().begin(),
                                            value.at_list_string().val().end()};
      std::pair<mlir::Identifier, mlir::Attribute> kv =
          b.getNamedAttr(name, b.getStrArrayAttr(r_vec));
      attr_vec.emplace_back(kv);
    }
    else if (value.has_at_data_type()) {
      auto dt = ::mlir::oneflow::symbolizeDataType(value.at_data_type());
      auto dt_str =
          ::mlir::oneflow::stringifyEnum(dt.getValueOr(::mlir::oneflow::DataType::InvalidDataType));
      std::pair<mlir::Identifier, mlir::Attribute> kv =
          b.getNamedAttr(name, b.getStringAttr(dt_str));
      attr_vec.emplace_back(kv);
    }
    else {
      module.emitError("can't handle user op attr: " + name);
      return failure();
    }
  }

  std::vector<NamedAttribute> inputs;
  for (auto input : op.user_conf().input()) {
    std::vector<llvm::StringRef> lbns = {input.second.s().begin(), input.second.s().end()};
    inputs.push_back(b.getNamedAttr(input.first, b.getStrArrayAttr(lbns)));
  }
  attr_vec.push_back(b.getNamedAttr("input", b.getDictionaryAttr(inputs)));

  std::vector<NamedAttribute> outputs;
  for (auto output : op.user_conf().output()) {
    std::vector<llvm::StringRef> lbns = {output.second.s().begin(), output.second.s().end()};
    outputs.push_back(b.getNamedAttr(output.first, b.getStrArrayAttr(lbns)));
  }
  attr_vec.push_back(b.getNamedAttr("output", b.getDictionaryAttr(outputs)));

  attr_vec.push_back(
      b.getNamedAttr("op_type_name", b.getStringAttr(op.user_conf().op_type_name())));

  attr_vec.push_back(b.getNamedAttr("name", b.getStringAttr(op.name())));

  return success();
}

LogicalResult Importer::operandsFromUserOp(const ::oneflow::OperatorConf &op,
                                           std::vector<Value> &operand_vec) {
  if (op.has_user_conf() == false) {
    module.emitError("Not a user op. op name: " + op.name());
    return failure();
  }
  for (auto kv : op.user_conf().input()) {
    // TODO: declare tensor containing field lbi
    for (int i = 0; i < kv.second.s_size(); i++) {
      const std::string &lbn = kv.second.s(i);
      if (lbn2result.find(lbn) != lbn2result.end()) {
        auto v = lbn2result.at(lbn);
        operand_vec.push_back(v);
      } else {
        // TODO: add placehorder ops for tick inputs
      }
    }
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
  const std::string &device_tag = pc.device_tag();
  std::vector<llvm::StringRef> device_vec = {pc.device_name().begin(), pc.device_name().end()};
  std::vector<NamedAttribute> attr_vec;
  attr_vec.push_back(b.getNamedAttr("device", b.getStringAttr(device_tag)));
  attr_vec.push_back(b.getNamedAttr("placement", b.getStrArrayAttr(device_vec)));
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

  for (auto kv : op.user_conf().output()) {
    int i = 0;
    for (const std::string &lbn : kv.second.s()) {
      lbn2result.insert({lbn, created_op->getResult(i)});
      i++;
    }
  }

  return success();
}  // namespace

LogicalResult Importer::processSystemOp(const ::oneflow::OperatorConf &op) {
  if (op.has_user_conf()) {
    module.emitError("Not a sys op. op name: " + op.name());
    return failure();
  }
  auto ilbis = job_wrapper.InputLbns4OpName(op.name());
  auto olbis = job_wrapper.InputLbns4OpName(op.name());
  job_wrapper.OutputLbns4OpName(op.name());
  std::vector<NamedAttribute> attr_vec;
  attr_vec.push_back(b.getNamedAttr(
      "input", b.getStrArrayAttr(std::vector<llvm::StringRef>({ilbis.begin(), ilbis.end()}))));
  attr_vec.push_back(b.getNamedAttr(
      "output", b.getStrArrayAttr(std::vector<llvm::StringRef>({olbis.begin(), olbis.end()}))));
  OperationState state(unknownLoc, "oneflow.system");
  attr_vec.push_back(b.getNamedAttr("op_name", b.getStringAttr(op.name())));
  state.addAttributes(attr_vec);
  std::vector<::mlir::Value> operand_vec;
  for (auto ilbi : ilbis) {
    if (lbn2result.find(ilbi) != lbn2result.end()) {
      auto v = lbn2result.at(ilbi);
      operand_vec.push_back(v);
    }
  }
  auto out_types = llvm::SmallVector<Type, 8>();
  for (auto olbi : olbis) { out_types.append({RankedTensorType::get({}, b.getF32Type())}); }
  state.addOperands(operand_vec);
  state.addTypes(out_types);
  auto created_op = b.createOperation(state);
  for (auto olbi : olbis) {
    int i = 0;
    lbn2result.insert({olbi, created_op->getResult(i)});
    i++;
  }
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

LogicalResult Importer::tryToUpdateJob() {
  std::cout << "try updating job\n";
  // TODO: add error handling
  auto convertOps = [](Operation *op) {
    if (op->hasAttr("op_type_name")) {
      oneflow::ReluOp defined_relu = llvm::dyn_cast<oneflow::ReluOp>(op);
      if (defined_relu) { defined_relu->dump(); }
      oneflow::ConstantOp defined_const = llvm::dyn_cast<oneflow::ConstantOp>(op);
      if (defined_const) { defined_const->dump(); }
    }
  };
  module.getBodyRegion().walk(convertOps);
  return success();
}

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
