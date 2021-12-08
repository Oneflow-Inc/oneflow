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
#include "oneflow/core/common/util.h"

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

namespace oneflow {

using PbMessage = google::protobuf::Message;

class JobImporter : Importer {
 public:
  JobImporter(RoundTripOneFlowJobWrapperInterface& job_wrapper, MLIRContext* context,
              ModuleOp module)
      : Importer(context, module), job_(job_wrapper.job()), job_wrapper_(job_wrapper) {}
  virtual ~JobImporter() = default;
  LogicalResult AppendDataInOperand(const std::string& lbn,
                                    std::vector<::mlir::Value>& operand_vec) override;
  LogicalResult AppendCtrlInOperand(const ::oneflow::OperatorConf& op,
                                    std::vector<::mlir::Value>& operand_vec) override;
  LogicalResult AddDeviceName(const ::oneflow::OperatorConf& op,
                              std::vector<NamedAttribute>& attr_vec) override;
  LogicalResult InsertOpResults(const ::oneflow::OperatorConf& op, Operation*) override;
  LogicalResult ProcessSystemOp(const ::oneflow::OperatorConf& op) override;
  LogicalResult ProcessJob();
  LogicalResult TryToUpdateJob();
  Type GetTensorTypeOfLbn(const std::string& lbn) override;

 private:
  std::unordered_map<std::string, mlir::OpResult> lbn2result_;
  std::unordered_map<std::string, mlir::OpResult> op_name2ctrl_result_;
  const ::oneflow::Job* job_;
  RoundTripOneFlowJobWrapperInterface& job_wrapper_;
};

LogicalResult JobImporter::AppendCtrlInOperand(const ::oneflow::OperatorConf& op,
                                               std::vector<::mlir::Value>& operand_vec) {
  for (auto& ctrl_in_op_name : op.ctrl_in_op_name()) {
    auto it = op_name2ctrl_result_.find(ctrl_in_op_name);
    if (it == op_name2ctrl_result_.end()) {
      GetModule().emitError("IR result not found for ctrl in op: " + ctrl_in_op_name);
      return failure();
    } else {
      operand_vec.push_back(it->second);
    }
  }
  return success();
}

LogicalResult JobImporter::AppendDataInOperand(const std::string& lbn,
                                               std::vector<::mlir::Value>& operand_vec) {
  auto it = lbn2result_.find(lbn);
  if (it == lbn2result_.end()) {
    GetModule().emitError("IR result not found for: " + lbn);
    return failure();
  } else {
    operand_vec.push_back(it->second);
    return success();
  }
}

LogicalResult JobImporter::InsertOpResults(const ::oneflow::OperatorConf& op,
                                           Operation* created_op) {
  auto output_lbns =
      created_op->getAttrOfType<ArrayAttr>(OpTrait::IsImportCompatible<void>::getOutputLBNsAttr());
  auto data_results = GetDataOutputResults(created_op);
  if (output_lbns.size() != data_results.size()) {
    output_lbns.dump();
    llvm::errs() << "output_lbns size: " << output_lbns.size()
                 << " != data_results size: " << data_results.size() << "\n"
                 << op.DebugString();
    created_op->getAttrDictionary().dump();
    created_op->dump();
    return failure();
  }
  for (const auto& data_out : llvm::enumerate(data_results)) {
    auto data_out_index = data_out.index();
    lbn2result_.insert({output_lbns[data_out_index].dyn_cast<StringAttr>().getValue().str(),
                        data_out.value().dyn_cast<OpResult>()});
  }
  if (auto ctrl_out = GetCtrlOutputResult(created_op)) {
    op_name2ctrl_result_.insert(
        {created_op->getAttrOfType<StringAttr>(OpTrait::IsOpConfCompatible<void>::getOpNameAttr())
             .getValue()
             .str(),
         ctrl_out->dyn_cast<OpResult>()});
  }
  return success();
}

LogicalResult JobImporter::AddDeviceName(const ::oneflow::OperatorConf& op,
                                         std::vector<NamedAttribute>& attr_vec) {
  const ::oneflow::ParallelConf& pc = job_wrapper_.ParallelConf4OpName(op.name());
  std::vector<llvm::StringRef> device_vec = {pc.device_name().begin(), pc.device_name().end()};
  attr_vec.push_back(
      GetBuilder().getNamedAttr(OpTrait::IsOpConfCompatible<void>::getDeviceNameAttr(),
                                GetBuilder().getStrArrayAttr(device_vec)));
  if (pc.has_hierarchy()) {
    attr_vec.push_back(GetBuilder().getNamedAttr(
        OpTrait::IsOpConfCompatible<void>::getHierarchyAttr(),
        GetBuilder().getI64ArrayAttr({pc.hierarchy().dim().begin(), pc.hierarchy().dim().end()})));
  }
  return success();
}

Type JobImporter::GetTensorTypeOfLbn(const std::string& lbn) {
  Type ret = this->GetBuilder().getNoneType();
  job_wrapper_.QueryLogicalBlob(
      lbn,
      [this, &ret](const int64_t* shape_begin, const int64_t* shape_end, ::oneflow::DataType dt) {
        if (auto t = this->GetTypeFromOneFlowDataType(dt)) {
          ret = RankedTensorType::get(ArrayRef<int64_t>(shape_begin, shape_end), t.getValue());
        }
      });
  return ret;
}

LogicalResult JobImporter::ProcessSystemOp(const ::oneflow::OperatorConf& op) {
  if (op.has_user_conf()) {
    GetModule().emitError("Not a sys op. op name: " + op.name());
    return failure();
  }
  auto input_bns_lbns = job_wrapper_.InputBns4OpName(op.name());
  auto input_bns = input_bns_lbns.first;
  auto input_lbns = input_bns_lbns.second;
  auto output_lbns = job_wrapper_.OutputLbns4OpName(op.name());
  job_wrapper_.OutputLbns4OpName(op.name());
  std::vector<NamedAttribute> attr_vec;
  if (failed(AddOpConf(op, attr_vec))) { return failure(); }
  if (failed(AddDeviceName(op, attr_vec))) { return failure(); }
  attr_vec.push_back(GetBuilder().getNamedAttr(
      "input_bns", GetBuilder().getStrArrayAttr(
                       std::vector<llvm::StringRef>({input_bns.begin(), input_bns.end()}))));
  attr_vec.push_back(GetBuilder().getNamedAttr(
      OpTrait::IsImportCompatible<void>::getOutputLBNsAttr(),
      GetBuilder().getStrArrayAttr(
          std::vector<llvm::StringRef>({output_lbns.begin(), output_lbns.end()}))));
  OperationState state(FileLineColLoc::get(GetMLIRContext(), op.name(), 0, 0),
                       oneflow::SystemOp::getOperationName());
  attr_vec.push_back(
      GetBuilder().getNamedAttr("op_type_case", GetBuilder().getI32IntegerAttr(op.op_type_case())));
  if (failed(AddOperandSegmentSizes(static_cast<int>(input_lbns.size()), op.ctrl_in_op_name_size(),
                                    attr_vec))) {
    return failure();
  }
  if (failed(AddResultSegmentSizes(output_lbns.size(), attr_vec))) { return failure(); }
  state.addAttributes(attr_vec);
  std::vector<::mlir::Value> operand_vec;
  for (const auto& input_lbn : input_lbns) {
    if (failed(AppendDataInOperand(input_lbn, operand_vec))) { return failure(); }
  }
  if (failed(AppendCtrlInOperand(op, operand_vec))) { return failure(); }
  auto out_types = llvm::SmallVector<Type, 8>();
  for (const auto& output_lbn : output_lbns) {
    out_types.push_back(GetTensorTypeOfLbn(output_lbn));
  }
  if (failed(AppendCtrlOutType(out_types))) { return failure(); }
  state.addOperands(operand_vec);
  state.addTypes(out_types);
  if (auto created_op = GetBuilder().createOperation(state)) {
    if (failed(InsertOpResults(op, created_op))) { return failure(); }
  } else {
    GetModule()->emitError("fail to create op, name: " + op.name());
    return failure();
  }
  return success();
}

LogicalResult JobImporter::ProcessJob() {
  auto func_type = GetBuilder().getFunctionType(llvm::None, llvm::None);
  auto function = mlir::FuncOp::create(GetRootLocation(), job_->job_conf().job_name(), func_type);
  auto& entryBlock = *function.addEntryBlock();
  GetBuilder().setInsertionPointToStart(&entryBlock);

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
  if (!returnOp) { GetBuilder().create<ReturnOp>(GetRootLocation()); }
  GetModule().push_back(function);
  return success();
}

template<typename OpType, typename AdaptorType>
void UpdatePlacement(OpType* op, AdaptorType& adaptor, ::oneflow::Job& job) {
  auto* pg = job.mutable_placement()->add_placement_group();
  pg->mutable_op_set()->add_op_name(adaptor.op_name().getValue().str());
  pg->mutable_parallel_conf()->set_device_tag(adaptor.device_tag().getValue().str());
  for (auto p : adaptor.device_name()) {
    pg->mutable_parallel_conf()->add_device_name(
        p.template dyn_cast<StringAttr>().getValue().str());
  }
  if (adaptor.hierarchy()) {
    for (auto dim : adaptor.hierarchy()) {
      pg->mutable_parallel_conf()->mutable_hierarchy()->add_dim(
          dim.template dyn_cast<IntegerAttr>().getInt());
    }
  }
}

LogicalResult JobImporter::TryToUpdateJob() {
  auto new_job = ::oneflow::Job();
  new_job.CopyFrom(*job_);
  new_job.clear_net();
  new_job.mutable_placement()->clear_placement_group();
  auto convertOps = [&](Operation* op) {
    if (llvm::dyn_cast<oneflow::SystemOp>(op)) {
      oneflow::SystemOpAdaptor system_op_adaptor(op->getOperands(), op->getAttrDictionary());
      UpdatePlacement(op, system_op_adaptor, new_job);
      auto op_name = system_op_adaptor.op_name().getValue().str();
      ::oneflow::OperatorConf op_conf = job_wrapper_.OpConf4OpName(op_name);
      for (const auto& ibn : llvm::enumerate(op->getAttrOfType<ArrayAttr>("input_bns"))) {
        auto result = GetDataInputOperands(op)[ibn.index()].dyn_cast<OpResult>();
        std::string new_val = GetOutputLbn(result).getValue();
        job_wrapper_.ReplaceInputLbnInOpCustomizedConf(
            &op_conf, ibn.value().dyn_cast<StringAttr>().getValue().str(), new_val);
      }
      if (succeeded(ConvertCtrlInputs(op, op_conf))) {
        *(new_job.mutable_net()->add_op()) = op_conf;
      } else {
        return WalkResult::interrupt();
      }
    } else if (llvm::dyn_cast<ReturnOp>(op) || llvm::dyn_cast<FuncOp>(op)
               || llvm::dyn_cast<ModuleOp>(op)) {
      return WalkResult::advance();
    } else {
      oneflow::UserOpAdaptor user_op_adaptor(op->getOperands(), op->getAttrDictionary());
      UpdatePlacement(op, user_op_adaptor, new_job);
      ::oneflow::OperatorConf op_conf;
      const std::string op_name = user_op_adaptor.op_name().getValue().str();
      auto user_conf = op_conf.mutable_user_conf();
      if (succeeded(ConvertUserOpInputs(op, user_op_adaptor, user_conf))
          && succeeded(ConvertUserOpOutputs(op, user_op_adaptor, user_conf))
          && succeeded(ConvertUserOpAttributes(op, user_op_adaptor, op_conf))
          && succeeded(ConvertCtrlInputs(op, op_conf))) {
        *(new_job.mutable_net()->add_op()) = op_conf;
      } else {
        return WalkResult::interrupt();
      }
    } /* convert op conf */
    return WalkResult::advance();
  };
  SymbolTable symbol_table(GetModule());
  if (symbol_table.lookup(job_wrapper_.job()->job_conf().job_name())
          ->walk(convertOps)
          .wasInterrupted()) {
    return failure();
  } else {
    job_wrapper_.UpdateJob(&new_job);
  }
  return success();
}

LogicalResult ApplyRoundTripPatterns(RoundTripOneFlowJobWrapperInterface& job_wrapper,
                                     MLIRContext* context, OwningModuleRef& module) {
  mlir::PassManager pm(context);
  pm.addNestedPass<mlir::FuncOp>(::mlir::createCanonicalizerPass());
  std::string graphviz;
  if (job_wrapper.IsLastIRPass() && std::getenv("ONEFLOW_MLIR_ENABLE_CODEGEN_FUSERS") != nullptr) {
    pm.addPass(oneflow::createOutlineJitFunctionPass());
  }
  pm.addNestedPass<mlir::FuncOp>(oneflow::createFuseIntoExistingOpPass());
  pm.addNestedPass<mlir::FuncOp>(::mlir::createCanonicalizerPass());
  llvm::raw_string_ostream os_graphviz(graphviz);
  pm.addPass(createPrintOpGraphPass(os_graphviz));
  if (mlir::failed(pm.run(*module))) {
    module->emitError("Failed to run round-trip passes");
    return failure();
  }
  job_wrapper.DumpLog("RoundTripOneFlowJob.mlir.dot", graphviz);
  std::string mlir;
  llvm::raw_string_ostream os_mlir(mlir);
  module->print(os_mlir);
  job_wrapper.DumpLog("RoundTripOneFlowJob.mlir", mlir);
  return success();
}

OwningModuleRef TranslateOneFlowJobToModule(llvm::StringRef str, MLIRContext* context) {
  std::string cpp_str = str.str();
  ::oneflow::Job job;
  google::protobuf::TextFormat::ParseFromString(cpp_str, &job);
  context->loadDialect<oneflow::OneFlowDialect>();
  context->loadDialect<StandardOpsDialect>();
  OwningModuleRef module(
      ModuleOp::create(FileLineColLoc::get(context, "", /*line=*/0, /*column=*/0)));
  return module;
}

void RoundTripOneFlowJob(
    RoundTripOneFlowJobWrapperInterface& job_wrapper,
    const std::function<bool(::oneflow::Job* job, std::string& reason)>& is_legit_job) {
  const ::oneflow::Job* job = job_wrapper.job();
  mlir::MLIRContext context;
  context.getOrLoadDialect<oneflow::OneFlowDialect>();
  context.loadDialect<StandardOpsDialect>();

  OwningModuleRef module(
      ModuleOp::create(FileLineColLoc::get(&context, "", /*line=*/0, /*column=*/0)));
  JobImporter imp(job_wrapper, &context, module.get());
  // TODO: Add flag in job desc to decide whether to run mlir optimizer
  if (succeeded(imp.ProcessJob())) {
    if (failed(ApplyRoundTripPatterns(job_wrapper, &context, module))) { exit(EXIT_FAILURE); }
    if (::oneflow::ParseBooleanFromEnv("ONEFLOW_MLIR_STDOUT", false)) {
      module->print(llvm::outs());
    }
    // TODO: Add flag in oneflow to define if failure in MLIR is allowed
    if (failed(imp.TryToUpdateJob())) {
      llvm::errs() << "fail to update job with IR, job will stay intact, job_name: "
                   << job->job_conf().job_name() << "\n";
      exit(EXIT_FAILURE);
    }

  } else {
    llvm::errs() << "fail to convert job to IR, job_name: " << job->job_conf().job_name() << "\n";
    exit(EXIT_FAILURE);
  }
}

void registerFromOneFlowJobTranslation() {
  TranslateToMLIRRegistration fromOneFlowJob("import-oneflow-job",
                                             [](llvm::StringRef str, MLIRContext* context) {
                                               return TranslateOneFlowJobToModule(str, context);
                                             });
}

}  // namespace oneflow

}  // namespace mlir
