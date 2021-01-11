#include "OneFlow/OneFlowOps.h"
#include "llvm-c/Core.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Module.h"
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
#include <iostream>
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

Value replaceGenericUserOp(mlir::PatternRewriter &rewriter,
                           ::mlir::Operation::operand_range operands, ::mlir::StringAttr op_name,
                           ::mlir::StringAttr op_type_name, ::mlir::DictionaryAttr attr) {
  std::cout << "replacing generic user op: " << op_type_name.getValue().str() << "\n";
  auto unknownLoc = FileLineColLoc::get("imported-protobuf", 0, 0, rewriter.getContext());
  mlir::Value created =
      rewriter.create<oneflow::ReluOp>(unknownLoc, operands[0], op_name).getResult();
  return created;
}

#include "OneFlow/OneFlowTranslateRewrites.inc"

using PbMessage = google::protobuf::Message;

class Importer {
 public:
  Importer(MLIRContext *context, ModuleOp module)
      : b(context),
        context(context),
        module(module),
        unknownLoc(FileLineColLoc::get("imported-protobuf", 0, 0, context)) {}
  LogicalResult processUserOp(const ::oneflow::OperatorConf &op);
  LogicalResult processJob(::oneflow::Job *job);

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
};

LogicalResult Importer::processUserOp(const ::oneflow::OperatorConf &op) {
  if (op.has_user_conf() == false) {
    module.emitError("Not a user op. op name: " + op.name());
    return failure();
  }
  const ::oneflow::UserOpConf &user_conf = op.user_conf();
  const std::string &type_name = user_conf.op_type_name();
  if (type_name == "constant") {
    if (user_conf.attr().at("is_floating_value").at_bool()) {
      mlir::Value created = b.create<oneflow::ConstantOp>(
                                 unknownLoc, user_conf.attr().at("floating_value").at_double())
                                .getResult();
      const std::string &lbn = user_conf.output().at("out").s(0);
      lbn2result.insert({lbn, created});
    } else {
      // b.create<ConstantOp>(unknownLoc, user_conf.attr().at("integer_value").at_int64());
    }
    return success();
  } else {
    std::vector<NamedAttribute> named_attr_vec;
    for (const class google::protobuf::MapPair<class std::basic_string<char>,
                                               class ::oneflow::AttrValue> &attr :
         op.user_conf().attr()) {
      const std::string &name = attr.first;
      const ::oneflow::AttrValue &value = attr.second;
      if (value.has_at_int32()) {
        std::pair<mlir::Identifier, mlir::Attribute> kv =
            b.getNamedAttr(name, b.getI32IntegerAttr(value.at_int32()));
        named_attr_vec.emplace_back(kv);
      }
#define DEFINE_ONE_ELIF(at_key, get_attr)                 \
  else if (value.has_##at_key()) {                        \
    std::pair<mlir::Identifier, mlir::Attribute> kv =     \
        b.getNamedAttr(name, b.get_attr(value.at_key())); \
    named_attr_vec.emplace_back(kv);                      \
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
    named_attr_vec.emplace_back(kv);                                                       \
  }
      DEFINE_ONE_ELIF(at_shape, getI64ArrayAttr, dim)
      DEFINE_ONE_ELIF(at_list_int32, getI32ArrayAttr, val)
      DEFINE_ONE_ELIF(at_list_int64, getI64ArrayAttr, val)
      DEFINE_ONE_ELIF(at_list_float, getF32ArrayAttr, val)
#undef DEFINE_ONE_ELIF
      else if (value.has_at_list_string()) {
        auto s_vec = std::vector<std::string>(
            {value.at_list_string().val().begin(), value.at_list_string().val().end()});
        auto r_vec = std::vector<llvm::StringRef>();
        for (auto s : s_vec) { r_vec.push_back(s); }
        std::pair<mlir::Identifier, mlir::Attribute> kv =
            b.getNamedAttr(name, b.getStrArrayAttr(r_vec));
        named_attr_vec.emplace_back(kv);
      }
      else if (value.has_at_data_type()) {
        auto dt = ::mlir::oneflow::symbolizeDataType(value.at_data_type());
        auto dt_str = ::mlir::oneflow::stringifyEnum(
            dt.getValueOr(::mlir::oneflow::DataType::InvalidDataType));
        std::pair<mlir::Identifier, mlir::Attribute> kv =
            b.getNamedAttr(name, b.getStringAttr(dt_str));
        named_attr_vec.emplace_back(kv);
      }
      else {
        module.emitError("can't handle user op attr: " + name);
        return failure();
      }
    }
    ArrayRef<NamedAttribute> named_attributes(named_attr_vec);

    std::vector<::mlir::Value> vs;
    for (auto kv : op.user_conf().input()) {
      // TODO: declare tensor containing field lbi
      // const std::string &ibn = kv.first;
      for (const std::string &lbn : kv.second.s()) {
        if (lbn2result.find(lbn) != lbn2result.end()) {
          auto v = lbn2result.at(lbn);
          vs.push_back(v);
        } else {
          // TODO: add placehorder tensors for tick inputs
        }
      }
    }
    auto out_types = llvm::SmallVector<Type, 8>();
    for (auto kv : op.user_conf().output()) {
      for (const std::string &lbn : kv.second.s()) {
        out_types.append({RankedTensorType::get({}, b.getF32Type())});
      }
    }
    ::mlir::ValueRange operands(vs);
    auto created = b.create<oneflow::UserOp>(unknownLoc, out_types, operands, op.name(),
                                             op.user_conf().op_type_name(),
                                             b.getDictionaryAttr(named_attributes));
    for (auto kv : op.user_conf().output()) {
      // const std::string &obn = kv.first;
      for (const std::string &lbn : kv.second.s()) {
        lbn2result.insert({lbn, created.getResults().back()});
      }
    }

    return success();
  }
}  // namespace

LogicalResult Importer::processJob(::oneflow::Job *job) {
  auto func_type = b.getFunctionType(llvm::None, llvm::None);
  auto function = mlir::FuncOp::create(unknownLoc, job->job_conf().job_name(), func_type);
  auto &entryBlock = *function.addEntryBlock();
  b.setInsertionPointToStart(&entryBlock);

  for (size_t i = 0; i < job->net().op_size(); i++) {
    ::oneflow::OperatorConf op = job->net().op(i);
    if (op.has_user_conf()) {
      std::cout << "processing user op: " << op.name() << "\n";
      std::cout << op.DebugString() << "\n";
      if (failed(processUserOp(op))) { return failure(); }
    }
  }
  ReturnOp returnOp;
  if (!entryBlock.empty()) { returnOp = dyn_cast<ReturnOp>(entryBlock.back()); }
  if (!returnOp) { b.create<ReturnOp>(unknownLoc); }
  module.push_back(function);
  return success();
}

OwningModuleRef translateOneFlowJobToModule(llvm::StringRef str, MLIRContext *context) {
  std::string cpp_str = str.str();
  ::oneflow::Job job;
  google::protobuf::TextFormat::ParseFromString(cpp_str, &job);
  context->loadDialect<oneflow::OneFlowDialect>();
  context->loadDialect<StandardOpsDialect>();
  OwningModuleRef module(
      ModuleOp::create(FileLineColLoc::get("", /*line=*/0, /*column=*/0, context)));
  Importer imp(context, module.get());
  if (failed(imp.processJob(&job))) { return {}; }

  std::cout << "import:\n";
  module->dump();

  OwningRewritePatternList import_patterns;
  import_patterns.insert<replaceGenericUserOpWithDefinedOp>(context);
  auto applied = applyPatternsAndFoldGreedily(module.get(), std::move(import_patterns));
  if (failed(applied)) { module->emitError("Failed to rewrite user ops"); }

  std::cout << "optimized:\n";
  module->dump();

  std::cout << "to export:\n";
  OwningRewritePatternList export_patterns;
  export_patterns.insert<replaceReluOpWithGenericUserOp>(context);
  if (failed(applyPatternsAndFoldGreedily(module.get(), std::move(export_patterns)))) {
    module->emitError("Failed to export user ops");
  }

  return module;
}
}  // namespace

void registerFromOneFlowJobTranslation() {
  TranslateToMLIRRegistration fromOneFlowJob("import-oneflow-job",
                                             [](llvm::StringRef str, MLIRContext *context) {
                                               return translateOneFlowJobToModule(str, context);
                                             });
}

}  // namespace mlir
