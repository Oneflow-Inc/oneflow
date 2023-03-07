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

#include "OneFlow/OneFlowDataTypeConversion.h"
#include "OneFlow/UserOpReflection.h"
#include "OneFlow/Transform/AggregateOps.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/common/data_type.pb.h"
#include "oneflow/core/framework/user_op_conf.pb.h"
#include "oneflow/core/job/job.pb.h"
#include "oneflow/core/operator/op_conf.pb.h"
#include "oneflow/core/operator/interface_blob_conf.pb.h"
#include "oneflow/core/ep/cuda/cuda_stream.h"

#include "OneFlow/OneFlowDialect.h"
#include "OneFlow/OneFlowOps.h"
#include "OneFlow/OneFlowOpTraits.h"
#include "OneFlow/Passes.h"
#include "OneFlow/MLIROneFlowTranslation.h"
#include "OneFlow/OneFlowUtils.h"
#include "OneFlow/UserOpConversion.h"

#include "mlir/Dialect/Tosa/Transforms/Passes.h"
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
#include "mlir/IR/Visitors.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Tools/mlir-translate/MlirTranslateMain.h"
#include "mlir/Tools/mlir-translate/Translation.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Parser/Parser.h"

#include "llvm-c/Core.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/None.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"

#include <google/protobuf/text_format.h>

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

  LogicalResult ProcessJob();
  LogicalResult ProcessSystemOp(const ::oneflow::OperatorConf& op) override;
  LogicalResult ProcessVariableOp(const ::oneflow::OperatorConf& op);
  LogicalResult ProcessInputOp(const ::oneflow::OperatorConf& op_conf, Block* entry_block,
                               size_t& input_count);
  LogicalResult ProcessOutputOp(const ::oneflow::OperatorConf& op_conf);

  LogicalResult TryToUpdateJob();
  LogicalResult ConvertUserOp(Operation* op, ::oneflow::Job& job);
  LogicalResult ConvertSystemOp(Operation* op, ::oneflow::Job& job);
  LogicalResult ConvertVariableOp(VariableOp op, ::oneflow::Job& job);
  LogicalResult ConvertInputOp(InputOp op, ::oneflow::Job& job);
  LogicalResult ConvertOutputOp(OutputOp op, ::oneflow::Job& job);

  Type GetTensorTypeOfLbn(const std::string& lbn) override;
  Type GetInterfaceBlobConfType(const ::oneflow::InterfaceBlobConf& blob_conf);

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
      GetModule().emitError("ctrl edge result of this op not found: " + ctrl_in_op_name
                            + ". op being controlled: " + op.name());
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
  Type ret{};
  job_wrapper_.QueryLogicalBlob(
      lbn, [this, &ret, &lbn](const int64_t* shape_begin, const int64_t* shape_end,
                              ::oneflow::DataType dt) {
        if (auto t = getTypeFromOneFlowDataType(GetMLIRContext(), dt)) {
          ret = RankedTensorType::get(ArrayRef<int64_t>(shape_begin, shape_end), t);
        } else {
          llvm::errs() << "fail to get data tensor type for: " << lbn << "\n";
        }
      });
  return ret;
}

LogicalResult JobImporter::ProcessSystemOp(const ::oneflow::OperatorConf& op) {
  if (op.has_user_conf()) {
    GetModule().emitError("Not a sys op. op name: " + op.name());
    return failure();
  }
  if (op.has_variable_conf()) { return ProcessVariableOp(op); }

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
                       SystemOp::getOperationName());
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
  if (auto created_op = GetBuilder().create(state)) {
    if (failed(InsertOpResults(op, created_op))) { return failure(); }
  } else {
    GetModule()->emitError("fail to create op, name: " + op.name());
    return failure();
  }
  return success();
}

LogicalResult JobImporter::ProcessVariableOp(const ::oneflow::OperatorConf& op_conf) {
  if (!op_conf.has_variable_conf()) {
    GetModule().emitError("Not a variable op. op name: " + op_conf.name());
    return failure();
  }

  if (op_conf.variable_conf().has_tick()) {
    GetModule().emitError("variable op has tick input. op name: " + op_conf.name());
    return failure();
  }

  OperationState state(FileLineColLoc::get(GetMLIRContext(), op_conf.name(), 0, 0),
                       "oneflow.variable");
  // attrs
  std::vector<NamedAttribute> attr_vec;
  if (failed(AddOpConf(op_conf, attr_vec))) { return failure(); }
  if (failed(AddDeviceName(op_conf, attr_vec))) { return failure(); }
  // attr output_lbns
  auto output_lbns_attr = GetBuilder().getStrArrayAttr({op_conf.name() + "/out"});
  attr_vec.emplace_back(GetBuilder().getNamedAttr(
      OpTrait::IsImportCompatible<void>::getOutputLBNsAttr(), output_lbns_attr));
  // attr shape
  auto shape_attr = GetAttrFromShape(op_conf.variable_conf().shape());
  auto shape_named_attr =
      GetBuilder().getNamedAttr(OpTrait::TensorSource<void>::getShapeAttrName(), shape_attr);
  attr_vec.emplace_back(shape_named_attr);
  // attr data_type
  if (op_conf.variable_conf().has_data_type()) {
    attr_vec.emplace_back(GetBuilder().getNamedAttr(
        OpTrait::TensorSource<void>::getDataTypeAttrName(),
        GetDataTypeAttr(GetMLIRContext(), op_conf.variable_conf().data_type()).getValue()));
  }
  // attr model_name
  if (op_conf.variable_conf().has_model_name()) {
    const std::string& model_name = op_conf.variable_conf().model_name();
    attr_vec.emplace_back(
        GetBuilder().getNamedAttr("model_name", GetBuilder().getStringAttr(model_name)));
  }
  // attr l1 l2 regularization
  if (op_conf.variable_conf().has_regularizer()
      && op_conf.variable_conf().regularizer().has_l1_l2_conf()) {
    if (op_conf.variable_conf().regularizer().l1_l2_conf().has_l1()) {
      float l1_regularization = op_conf.variable_conf().regularizer().l1_l2_conf().l1();
      attr_vec.emplace_back(GetBuilder().getNamedAttr(
          "l1_regularization", GetBuilder().getF32FloatAttr(l1_regularization)));
    }
    if (op_conf.variable_conf().regularizer().l1_l2_conf().has_l2()) {
      float l2_regularization = op_conf.variable_conf().regularizer().l1_l2_conf().l2();
      attr_vec.emplace_back(GetBuilder().getNamedAttr(
          "l2_regularization", GetBuilder().getF32FloatAttr(l2_regularization)));
    }
  }
  // attr trainable
  if (op_conf.variable_conf().has_trainable()) {
    bool trainable = op_conf.variable_conf().trainable();
    attr_vec.emplace_back(
        GetBuilder().getNamedAttr("trainable", GetBuilder().getBoolAttr(trainable)));
  }
  if (op_conf.variable_conf().has_initializer()) {
    if (op_conf.variable_conf().initializer().has_constant_conf()) {
      const mlir::Attribute const_initialize_attr = GetBuilder().getF32FloatAttr(
          op_conf.variable_conf().initializer().constant_conf().value());
      attr_vec.emplace_back(GetBuilder().getNamedAttr("float_initializer", const_initialize_attr));
    } else if (op_conf.variable_conf().initializer().has_constant_int_conf()) {
      const mlir::Attribute const_initialize_attr =
          getSI64IntegerAttr(op_conf.variable_conf().initializer().constant_int_conf().value());
      attr_vec.emplace_back(
          GetBuilder().getNamedAttr("integer_initializer", const_initialize_attr));
    }
  }
  // attr parallel
  auto conf = this->job_wrapper_.ParallelConf4OpName(op_conf.name());

  auto nd_size = conf.hierarchy().dim().size();
  auto nd_sbp = op_conf.variable_conf().nd_sbp();
  auto parallel = mlir::oneflow::SBPTranslation::ConvertNdSbpToPsig(
      GetBuilder(), std::vector<std::string>(nd_sbp.begin(), nd_sbp.end()), nd_size);
  attr_vec.emplace_back(
      GetBuilder().getNamedAttr(OpTrait::TensorSource<void>::getSbpAttrName(), parallel));
  // add attrs
  state.addAttributes(attr_vec);
  // operands
  std::vector<::mlir::Value> operand_vec;
  if (failed(AppendCtrlInOperand(op_conf, operand_vec))) { return failure(); }
  state.addOperands(operand_vec);
  // result types
  llvm::SmallVector<Type, 8> out_types;
  auto output_lbn = op_conf.name() + "/out";
  out_types.push_back(GetTensorTypeOfLbn(output_lbn));
  if (failed(AppendCtrlOutType(out_types))) { return failure(); }
  state.addTypes(out_types);
  SetOpStateLoc(op_conf, state);
  // create op
  auto op = GetBuilder().create(state);
  if (!op) {
    GetModule()->emitError("fail to create op, name: " + op_conf.name());
    return failure();
  }
  // record result
  if (op->getNumResults() != 2) {
    op->emitError("variable op should has two results (out and ctrl_output), but got "
                  + std::to_string(op->getNumResults()) + "\n");
    return failure();
  }
  if (!lbn2result_.emplace(output_lbn, op->getResult(0)).second) {
    op->emitError("lbn already exists, lbn: ") << output_lbn;
    return failure();
  }
  if (!op_name2ctrl_result_.emplace(op_conf.name(), op->getResult(1)).second) {
    op->emitError("ctrl output already exists, op_name: ") << op_conf.name();
    return failure();
  }
  return success();
}

LogicalResult JobImporter::ProcessInputOp(const ::oneflow::OperatorConf& op_conf,
                                          Block* entry_block, size_t& input_count) {
  if (!op_conf.has_input_conf()) {
    GetModule().emitError("Not a input op. op name: " + op_conf.name());
    return failure();
  }

  if (op_conf.input_conf().has_tick()) {
    GetModule().emitError("input op has tick input. op name: " + op_conf.name());
    return failure();
  }

  OperationState state(FileLineColLoc::get(GetMLIRContext(), op_conf.name(), 0, 0),
                       "oneflow.input");
  // attrs
  std::vector<NamedAttribute> attr_vec;
  if (failed(AddOpConf(op_conf, attr_vec))) { return failure(); }
  if (failed(AddDeviceName(op_conf, attr_vec))) { return failure(); }
  // attr output_lbns
  auto output_lbns_attr = GetBuilder().getStrArrayAttr({op_conf.name() + "/out"});
  attr_vec.emplace_back(GetBuilder().getNamedAttr(
      OpTrait::IsImportCompatible<void>::getOutputLBNsAttr(), output_lbns_attr));
  // attr shape
  if (op_conf.input_conf().blob_conf().has_shape()) {
    auto shape_attr = GetAttrFromShape(op_conf.input_conf().blob_conf().shape());
    attr_vec.emplace_back(
        GetBuilder().getNamedAttr(OpTrait::TensorSource<void>::getShapeAttrName(), shape_attr));
  }
  // attr data_type
  if (op_conf.input_conf().blob_conf().has_data_type()) {
    attr_vec.emplace_back(GetBuilder().getNamedAttr(
        OpTrait::TensorSource<void>::getDataTypeAttrName(),
        GetDataTypeAttr(GetMLIRContext(), op_conf.input_conf().blob_conf().data_type())
            .getValue()));
  }
  // attr is_dynamic
  if (op_conf.input_conf().blob_conf().has_is_dynamic()) {
    bool is_dynamic = op_conf.input_conf().blob_conf().is_dynamic();
    attr_vec.emplace_back(GetBuilder().getNamedAttr(
        OpTrait::TensorSource<void>::getIsDynamicAttrName(), GetBuilder().getBoolAttr(is_dynamic)));
  }
  // attr nd_sbp
  if (op_conf.input_conf().blob_conf().has_nd_sbp()) {
    auto nd_sbp_attr = ConvertNdSbpToAttr(GetBuilder(), op_conf.input_conf().blob_conf().nd_sbp());
    attr_vec.emplace_back(
        GetBuilder().getNamedAttr(OpTrait::TensorSource<void>::getNdSbpAttrName(), nd_sbp_attr));
  }
  // attr job_name
  if (op_conf.input_conf().has_job_name()) {
    const std::string& job_name = op_conf.input_conf().job_name();
    attr_vec.emplace_back(
        GetBuilder().getNamedAttr("job_name", GetBuilder().getStringAttr(job_name)));
  }
  // add attrs
  state.addAttributes(attr_vec);
  // operands
  std::vector<::mlir::Value> operand_vec;
  operand_vec.emplace_back(entry_block->getArgument(input_count++));
  if (failed(AppendCtrlInOperand(op_conf, operand_vec))) { return failure(); }
  state.addOperands(operand_vec);
  // result types
  llvm::SmallVector<Type, 8> out_types;
  auto output_lbn = op_conf.name() + "/out";
  out_types.push_back(GetTensorTypeOfLbn(output_lbn));
  if (failed(AppendCtrlOutType(out_types))) { return failure(); }
  state.addTypes(out_types);
  // create op
  auto op = GetBuilder().create(state);
  if (!op) {
    GetModule()->emitError("fail to create op, name: " + op_conf.name());
    return failure();
  }
  // record result
  if (op->getNumResults() != 2) {
    op->emitError("input op should has two results (out and ctrl_output), but got "
                  + std::to_string(op->getNumResults()) + "\n");
    return failure();
  }
  if (!lbn2result_.emplace(output_lbn, op->getResult(0)).second) {
    op->emitError("lbn already exists, lbn: ") << output_lbn;
    return failure();
  }
  if (!op_name2ctrl_result_.emplace(op_conf.name(), op->getResult(1)).second) {
    op->emitError("ctrl output already exists, op_name: ") << op_conf.name();
    return failure();
  }
  return success();
}

LogicalResult JobImporter::ProcessOutputOp(const ::oneflow::OperatorConf& op_conf) {
  if (!op_conf.has_output_conf()) {
    GetModule().emitError("Not a output op. op name: " + op_conf.name());
    return failure();
  }

  OperationState state(FileLineColLoc::get(GetMLIRContext(), op_conf.name(), 0, 0),
                       "oneflow.output");
  // attrs
  std::vector<NamedAttribute> attr_vec;
  if (failed(AddOpConf(op_conf, attr_vec))) { return failure(); }
  if (failed(AddDeviceName(op_conf, attr_vec))) { return failure(); }
  // attr output_lbns
  auto output_lbns_attr = GetBuilder().getStrArrayAttr({op_conf.name() + "/out"});
  attr_vec.emplace_back(GetBuilder().getNamedAttr(
      OpTrait::IsImportCompatible<void>::getOutputLBNsAttr(), output_lbns_attr));
  // attr shape
  if (op_conf.output_conf().blob_conf().has_shape()) {
    auto shape_attr = GetAttrFromShape(op_conf.output_conf().blob_conf().shape());
    attr_vec.emplace_back(
        GetBuilder().getNamedAttr(OpTrait::TensorSource<void>::getShapeAttrName(), shape_attr));
  }
  // attr data_type
  if (op_conf.output_conf().blob_conf().has_data_type()) {
    attr_vec.emplace_back(GetBuilder().getNamedAttr(
        OpTrait::TensorSource<void>::getDataTypeAttrName(),
        GetDataTypeAttr(GetMLIRContext(), op_conf.output_conf().blob_conf().data_type())
            .getValue()));
  }
  // attr is_dynamic
  if (op_conf.output_conf().blob_conf().has_is_dynamic()) {
    bool is_dynamic = op_conf.output_conf().blob_conf().is_dynamic();
    attr_vec.emplace_back(GetBuilder().getNamedAttr(
        OpTrait::TensorSource<void>::getIsDynamicAttrName(), GetBuilder().getBoolAttr(is_dynamic)));
  }
  // attr nd_sbp
  if (op_conf.output_conf().blob_conf().has_nd_sbp()) {
    auto nd_sbp_attr = ConvertNdSbpToAttr(GetBuilder(), op_conf.output_conf().blob_conf().nd_sbp());
    attr_vec.emplace_back(
        GetBuilder().getNamedAttr(OpTrait::TensorSource<void>::getNdSbpAttrName(), nd_sbp_attr));
  }
  // attr job_name
  if (op_conf.output_conf().has_job_name()) {
    const std::string& job_name = op_conf.output_conf().job_name();
    attr_vec.emplace_back(
        GetBuilder().getNamedAttr("job_name", GetBuilder().getStringAttr(job_name)));
  }
  // add attrs
  state.addAttributes(attr_vec);
  // operands
  std::vector<::mlir::Value> operand_vec;
  auto input_bns_lbns = job_wrapper_.InputBns4OpName(op_conf.name());
  if (input_bns_lbns.second.size() != 1) {
    GetModule()->emitError("output op should has only one input, op_name: " + op_conf.name());
    return failure();
  }
  if (failed(AppendDataInOperand(input_bns_lbns.second[0], operand_vec))) { return failure(); }
  if (failed(AppendCtrlInOperand(op_conf, operand_vec))) { return failure(); }
  state.addOperands(operand_vec);
  // result types
  llvm::SmallVector<Type, 8> out_types;
  auto output_lbn = op_conf.name() + "/out";
  out_types.push_back(GetTensorTypeOfLbn(output_lbn));
  if (failed(AppendCtrlOutType(out_types))) { return failure(); }
  state.addTypes(out_types);
  // create op
  auto op = GetBuilder().create(state);
  if (!op) {
    GetModule()->emitError("fail to create op, name: " + op_conf.name());
    return failure();
  }
  // record result
  if (op->getNumResults() != 2) {
    op->emitError("output_conf op should has two results (out and ctrl_output), but got "
                  + std::to_string(op->getNumResults()) + "\n");
    return failure();
  }
  if (!lbn2result_.emplace(output_lbn, op->getResult(0)).second) {
    op->emitError("lbn already exists, lbn: ") << output_lbn;
    return failure();
  }
  if (!op_name2ctrl_result_.emplace(op_conf.name(), op->getResult(1)).second) {
    op->emitError("ctrl output already exists, op_name: ") << op_conf.name();
    return failure();
  }
  return success();
}

LogicalResult JobImporter::ProcessJob() {
  llvm::SmallVector<Type, 8> input_types;
  llvm::SmallVector<Type, 4> result_types;
  llvm::SmallVector<Value, 4> results;
  bool is_succeeded = true;

  job_wrapper_.TopoForEachOpConf([&](const ::oneflow::OperatorConf* op_conf) {
    if (op_conf->has_input_conf()) {
      auto type = GetInterfaceBlobConfType(op_conf->input_conf().blob_conf());
      if (type) {
        input_types.emplace_back(type);
      } else {
        GetModule()->emitError("fail to collect func arg types for job:\n"
                               + op_conf->DebugString());
        is_succeeded = false;
      }
    }
  });
  if (!is_succeeded) { return failure(); }

  auto func_type = GetBuilder().getFunctionType(input_types, llvm::None);
  auto job_op =
      GetBuilder().create<oneflow::Job>(GetRootLocation(), job_->job_conf().job_name(), func_type);
  auto* entryBlock = job_op.addEntryBlock();
  GetBuilder().setInsertionPointToStart(entryBlock);

  is_succeeded = true;
  size_t input_count = 0;
  job_wrapper_.TopoForEachOpConf([&](const ::oneflow::OperatorConf* op_conf) {
    if (is_succeeded == false) { return; }
    if (op_conf->has_user_conf()) {
      is_succeeded = succeeded(ProcessUserOp(*op_conf));
    } else if (op_conf->has_input_conf()) {
      is_succeeded = succeeded(ProcessInputOp(*op_conf, entryBlock, input_count));
    } else if (op_conf->has_output_conf()) {
      is_succeeded = succeeded(ProcessOutputOp(*op_conf));
      if (is_succeeded) {
        auto result = entryBlock->back().getResult(0);
        results.emplace_back(result);
        result_types.emplace_back(result.getType());
      }
    } else {
      is_succeeded = succeeded(ProcessSystemOp(*op_conf));
    }
  });
  if (is_succeeded == false) { return failure(); }
  mlir::oneflow::ReturnOp return_op;
  if (!entryBlock->empty()) { return_op = dyn_cast<mlir::oneflow::ReturnOp>(entryBlock->back()); }
  if (!return_op) { GetBuilder().create<mlir::oneflow::ReturnOp>(GetRootLocation(), results); }

  func_type = GetBuilder().getFunctionType(input_types, result_types);
  job_op.getOperation()->setAttr(oneflow::Job::getTypeAttrName(), TypeAttr::get(func_type));
  GetModule().push_back(job_op);
  return success();
}

template<typename OpType, typename AdaptorType>
void UpdatePlacement(OpType* op, AdaptorType& adaptor, ::oneflow::Job& job) {
  auto* pg = job.mutable_placement()->add_placement_group();
  pg->mutable_op_set()->add_op_name(adaptor.op_name().str());
  pg->mutable_parallel_conf()->set_device_tag(adaptor.device_tag().str());
  for (auto p : adaptor.device_name()) {
    pg->mutable_parallel_conf()->add_device_name(
        p.template dyn_cast<StringAttr>().getValue().str());
  }
  if (::llvm::Optional<ArrayAttr> hierarchy = adaptor.hierarchy()) {
    for (auto dim : hierarchy->getValue()) {
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

  Operation* job_op = nullptr;
  llvm::SmallVector<Value, 4> outputs;

  auto find_first_job = [&](oneflow::Job job) -> WalkResult {
    job_op = job.getOperation();
    new_job.mutable_job_conf()->set_job_name(job.sym_name().str());
    return WalkResult::interrupt();
  };

  GetModule().getOperation()->walk(find_first_job);
  if (!job_op) {
    GetModule()->emitError("job not found. module op: ") << *GetModule();
    return failure();
  }

  auto ConvertOp = [&](Operation* op) -> WalkResult {
    if (op->hasTrait<OpTrait::IsOpConfCompatible>()) {
      if (llvm::dyn_cast<oneflow::UserOp>(op)) {
        if (failed(ConvertUserOp(op, new_job))) {
          op->emitError("failed to convert generic UserOp: ") << *op;
          return WalkResult::interrupt();
        }
      } else if (llvm::dyn_cast<oneflow::SystemOp>(op)) {
        if (failed(ConvertSystemOp(op, new_job))) {
          op->emitError("failed to convert SystemOp: ") << *op;
          return WalkResult::interrupt();
        }
      } else if (auto variable_op = llvm::dyn_cast<oneflow::VariableOp>(op)) {
        if (failed(ConvertVariableOp(variable_op, new_job))) {
          op->emitError("failed to process VariableOp: ") << *op;
          return WalkResult::interrupt();
        }
      } else if (llvm::dyn_cast<oneflow::InputOp>(op) || llvm::dyn_cast<oneflow::OutputOp>(op)) {
        // do nothing and advance
      } else {
        if (!dyn_cast<UserOpCompatible>(op)) {
          op->emitError("op is not UserOpCompatible ") << *op;
          return WalkResult::interrupt();
        }
        if (failed(ConvertUserOp(op, new_job))) {
          op->emitError("failed to process UserOp: ") << *op;
          return WalkResult::interrupt();
        }
      }
    } else if (llvm::dyn_cast<mlir::oneflow::Job>(op)) {
      // do nothing and advance
    } else if (op->hasTrait<OpTrait::OnlyExistsInIR>()) {
      // do nothing and advance
    } else if (auto return_op = llvm::dyn_cast<mlir::oneflow::ReturnOp>(op)) {
      for (auto operand : return_op->getOperands()) { outputs.emplace_back(operand); }
    } else {
      op->emitError("unexcepted op: ") << *op;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  };
  if (job_op->walk(ConvertOp).wasInterrupted()) { return failure(); }

  // add input op
  auto arguments = llvm::dyn_cast<oneflow::Job>(job_op).body().front().getArguments();
  for (BlockArgument argument : arguments) {
    for (auto& use : argument.getUses()) {
      Operation* owner = use.getOwner();
      if (auto input_op = dyn_cast<oneflow::InputOp>(owner)) {
        if (failed(ConvertInputOp(input_op, new_job))) { return failure(); }
      } else {
        return failure();
      }
    }
  }
  // add output op
  for (auto output : outputs) {
    Operation* owner = output.getDefiningOp();
    if (auto output_op = dyn_cast<oneflow::OutputOp>(owner)) {
      if (failed(ConvertOutputOp(output_op, new_job))) { return failure(); }
    } else {
      return failure();
    }
  }

  job_wrapper_.UpdateJob(&new_job);
  return success();
}

LogicalResult JobImporter::ConvertUserOp(Operation* op, ::oneflow::Job& job) {
  oneflow::ConfOpAdaptor conf_op_adaptor(op->getOperands(), op->getAttrDictionary());
  UpdatePlacement(op, conf_op_adaptor, job);
  StringRef op_name = conf_op_adaptor.op_name();

  auto* op_conf = job.mutable_net()->add_op();
  auto* user_conf = op_conf->mutable_user_conf();
  if (!succeeded(ConvertUserOpInputs(op, op_name, user_conf))) {
    op->emitError("fail to convert user op inputs");
    return failure();
  }
  if (!succeeded(ConvertUserOpOutputs(op, op_name, user_conf))) {
    op->emitError("fail to convert user op outputs");
    return failure();
  }
  if (!succeeded(user_op::ConvertUserOpAttributes(op, *op_conf, false))) {
    op->emitError("fail to convert user op attributes");
    return failure();
  }
  if (!succeeded(ConvertCtrlInputs(op, *op_conf))) {
    op->emitError("fail to convert user op control inputs");
    return failure();
  }
  return success();
}

LogicalResult JobImporter::ConvertSystemOp(Operation* op, ::oneflow::Job& job) {
  oneflow::SystemOpAdaptor system_op_adaptor(op->getOperands(), op->getAttrDictionary());
  UpdatePlacement(op, system_op_adaptor, job);
  auto op_name = system_op_adaptor.op_name().str();
  ::oneflow::OperatorConf op_conf = job_wrapper_.OpConf4OpName(op_name);
  for (const auto& ibn : llvm::enumerate(op->getAttrOfType<ArrayAttr>("input_bns"))) {
    auto result = GetDataInputOperands(op)[ibn.index()].dyn_cast<OpResult>();
    std::string new_val = user_op::GetOutputLbn(result).getValue();
    job_wrapper_.ReplaceInputLbnInOpCustomizedConf(
        &op_conf, ibn.value().dyn_cast<StringAttr>().getValue().str(), new_val);
  }
  if (failed(ConvertCtrlInputs(op, op_conf))) { return failure(); }
  *(job.mutable_net()->add_op()) = op_conf;
  return success();
}

LogicalResult JobImporter::ConvertVariableOp(VariableOp op, ::oneflow::Job& job) {
  oneflow::VariableOpAdaptor op_adaptor(op->getOperands(), op->getAttrDictionary());
  UpdatePlacement(&op, op_adaptor, job);
  auto* op_conf = job.mutable_net()->add_op();
  return ConvertVariableOpConf(op, op_conf);
}

LogicalResult JobImporter::ConvertInputOp(InputOp op, ::oneflow::Job& job) {
  oneflow::InputOpAdaptor op_adaptor(op->getOperands(), op->getAttrDictionary());
  UpdatePlacement(&op, op_adaptor, job);
  auto* op_conf = job.mutable_net()->add_op();
  return ConvertInputOpConf(op, op_conf);
}

LogicalResult JobImporter::ConvertOutputOp(OutputOp op, ::oneflow::Job& job) {
  oneflow::OutputOpAdaptor op_adaptor(op->getOperands(), op->getAttrDictionary());
  UpdatePlacement(&op, op_adaptor, job);
  auto* op_conf = job.mutable_net()->add_op();
  return ConvertOutputOpConf(op, op_conf);
}

Type JobImporter::GetInterfaceBlobConfType(const ::oneflow::InterfaceBlobConf& blob_conf) {
  if (!blob_conf.has_data_type()) { return Type{}; }
  if (!blob_conf.has_shape()) { return Type{}; };
  if (auto data_type = getTypeFromOneFlowDataType(GetMLIRContext(), blob_conf.data_type())) {
    return RankedTensorType::get({blob_conf.shape().dim().begin(), blob_conf.shape().dim().end()},
                                 data_type);
  } else {
    return Type{};
  }
}

void DumpMLIR(RoundTripOneFlowJobWrapperInterface& job_wrapper, ModuleOp module,
              const std::string& name) {
  std::string mlir;
  llvm::raw_string_ostream os_mlir(mlir);
  module->print(os_mlir);
  job_wrapper.DumpLog(name + ".mlir", mlir);
}

LogicalResult ApplyRoundTripPatterns(RoundTripOneFlowJobWrapperInterface& job_wrapper,
                                     MLIRContext* context, OwningOpRef<ModuleOp>& module) {
  mlir::PassManager pm(context);
  if (::oneflow::ParseBooleanFromEnv("ONEFLOW_MLIR_ENABLE_TIMING", false)) { pm.enableTiming(); }
  mlir::oneflow::CheckEnableIRPrinting(pm);
  // this canonicalizer should create concrete ops and create fuse opportunities
  pm.addPass(createCanonicalizerPass());
  if (job_wrapper.IsLastIRPass()
      && ::oneflow::ParseBooleanFromEnv("ONEFLOW_MLIR_ENABLE_CODEGEN_FUSERS", false)) {
    pm.addPass(oneflow::createOutlineJitFunctionPass());
  }
  // we must do auto nhwc and eliminate redundant transpose op first, avoid insert redundant
  // transpose op due to fuse pattern like normlazation_add_relu.
  pm.addPass(oneflow::createAutoNhwcPass());
  if (::oneflow::ParseBooleanFromEnv("ONEFLOW_MLIR_CSE", false)) {
    auto cse_state = std::make_shared<CSEState>();
    auto passes = createCSEPasses(cse_state);
    pm.addPass(std::move(passes.first));
    pm.addPass(createCSEPass());
    pm.addPass(std::move(passes.second));
  }
  if (job_wrapper.IsLastIRPass()
      && ::oneflow::ParseBooleanFromEnv("ONEFLOW_MLIR_FUSE_FORWARD_OPS", false)) {
    pm.addPass(oneflow::createFuseForwardOps());
    pm.addPass(oneflow::createFuseIntoExistingOpPass());
  }
  if (job_wrapper.IsLastIRPass()
      && ::oneflow::ParseBooleanFromEnv("ONEFLOW_MLIR_ENABLE_CODEGEN_FUSERS", false)) {
    pm.addPass(oneflow::createOutlineJitFunctionPass());
  }
  if (!job_wrapper.IsLastIRPass()
      && ::oneflow::ParseBooleanFromEnv("ONEFLOW_MLIR_FUSE_OPS_WITH_BACKWARD_IMPL", false)) {
    pm.addPass(oneflow::createFuseOpsWithBackwardImpl());
  }
  // TODO: support backward or put it in a env flag
  if (job_wrapper.IsLastIRPass()
      && ::oneflow::ParseBooleanFromEnv("ONEFLOW_MLIR_GROUP_MATMUL", false)) {
    pm.addPass(oneflow::createGroupMatMul());
  }
  if (!job_wrapper.IsLastIRPass()
      && ::oneflow::ParseBooleanFromEnv("ONEFLOW_MLIR_ENABLE_INFERENCE_OPTIMIZATION", false)) {
    pm.addPass(oneflow::createPreConvertInferenceOpPass());
    pm.addPass(oneflow::createConvertInferenceOpPass());
    pm.addPass(oneflow::createPostConvertInferenceOpPass());
  }
  if (!job_wrapper.IsLastIRPass()
      && ::oneflow::ParseBooleanFromEnv("ONEFLOW_MLIR_FUSE_NORMALIZATION_OPS", false)) {
    pm.addPass(oneflow::createFuseNormalizationOps());
  }
  if (job_wrapper.IsLastIRPass()
      && ::oneflow::ParseBooleanFromEnv("ONEFLOW_MLIR_FUSE_KERNEL_LAUNCH", false)) {
    pm.addPass(createAggregateComputeOpsPass());

    auto wrap_pass = createWrapOpsToKernelLaunchPass();
    std::string options =
        "mode="
        + (::oneflow::ParseBooleanFromEnv("ONEFLOW_KERNEL_ENABLE_CUDA_GRAPH", false)
               ? wrap_mode::CUDA_GRAPH
               : wrap_mode::SIMPLE);

    (void)wrap_pass->initializeOptions(options);
    pm.addPass(std::move(wrap_pass));
  }
  pm.addPass(createCanonicalizerPass());
  if (::oneflow::ParseBooleanFromEnv("ONEFLOW_MLIR_PRINT_STATS", false)) {
    pm.addPass(createPrintOpStatsPass());
  }
  std::string graphviz;
  llvm::raw_string_ostream os_graphviz(graphviz);
  const bool shouldPrintGraphviz =
      ::oneflow::ParseBooleanFromEnv("ONEFLOW_MLIR_PRINT_OP_GRAPH", false);
  if (shouldPrintGraphviz) { pm.addPass(createPrintOpGraphPass(os_graphviz)); }
  if (mlir::failed(pm.run(*module))) {
    module->emitError("Failed to run round-trip passes");
    return failure();
  }
  if (shouldPrintGraphviz) {
    job_wrapper.DumpLog("RoundTripOneFlowJob.optimized.mlir.dot", graphviz);
  }
  DumpMLIR(job_wrapper, module.get(), "RoundTripOneFlowJob.optimized");
  return success();
}

OwningOpRef<ModuleOp> TranslateOneFlowJobToModule(llvm::StringRef str, MLIRContext* context) {
  std::string cpp_str = str.str();
  ::oneflow::Job job;
  google::protobuf::TextFormat::ParseFromString(cpp_str, &job);
  context->loadDialect<oneflow::OneFlowDialect>();
  context->loadDialect<mlir::func::FuncDialect>();
  OwningOpRef<ModuleOp> module(
      ModuleOp::create(FileLineColLoc::get(context, "", /*line=*/0, /*column=*/0)));
  return module;
}

void RoundTripOneFlowJob(
    RoundTripOneFlowJobWrapperInterface& job_wrapper,
    const std::function<bool(::oneflow::Job* job, std::string& reason)>& is_legit_job) {
  const ::oneflow::Job* job = job_wrapper.job();
  mlir::MLIRContext context;
  context.getOrLoadDialect<oneflow::OneFlowDialect>();
  context.loadDialect<mlir::func::FuncDialect>();

  OwningOpRef<ModuleOp> module(
      ModuleOp::create(FileLineColLoc::get(&context, "", /*line=*/0, /*column=*/0)));
  JobImporter imp(job_wrapper, &context, module.get());
  // TODO: Add flag in job desc to decide whether to run mlir optimizer
  if (succeeded(imp.ProcessJob())) {
    DumpMLIR(job_wrapper, module.get(), "RoundTripOneFlowJob.imported");
    if (failed(ApplyRoundTripPatterns(job_wrapper, &context, module))) { exit(EXIT_FAILURE); }
    if (::oneflow::ParseBooleanFromEnv("ONEFLOW_MLIR_STDOUT", false)
        && job_wrapper.IsLastIRPass()) {
      // for FileCheck
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

std::string ConvertJobToTosaIR(RoundTripOneFlowJobWrapperInterface& job_wrapper) {
  const ::oneflow::Job* job = job_wrapper.job();
  mlir::MLIRContext context;
  context.getOrLoadDialect<oneflow::OneFlowDialect>();
  context.loadDialect<mlir::func::FuncDialect>();

  OwningOpRef<ModuleOp> module(
      ModuleOp::create(FileLineColLoc::get(&context, "", /*line=*/0, /*column=*/0)));
  JobImporter imp(job_wrapper, &context, module.get());
  if (succeeded(imp.ProcessJob())) {
    mlir::PassManager pm(&context);
    pm.addPass(createCanonicalizerPass());
    pm.addPass(createConvertToSignlessForTosaPass());
    pm.addPass(createLowerOneFlowToTosaPass());
    pm.addNestedPass<func::FuncOp>(tosa::createTosaMakeBroadcastablePass());
    if (mlir::failed(pm.run(*module))) {
      module->emitError("Failed to run oneflow-to-tosa pass");
      exit(EXIT_FAILURE);
    }

    std::string mlir;
    llvm::raw_string_ostream os_mlir(mlir);
    module->print(os_mlir);
    return mlir;
  } else {
    const auto& job_name = job->job_conf().job_name();
    llvm::errs() << "fail to convert job to IR, job_name: " << job_name << "\n";
    exit(EXIT_FAILURE);
  }
}

std::string ConvertJobToIR(RoundTripOneFlowJobWrapperInterface& job_wrapper) {
  const ::oneflow::Job* job = job_wrapper.job();
  mlir::MLIRContext context;
  context.getOrLoadDialect<oneflow::OneFlowDialect>();
  context.loadDialect<mlir::func::FuncDialect>();

  OwningOpRef<ModuleOp> module(
      ModuleOp::create(FileLineColLoc::get(&context, "", /*line=*/0, /*column=*/0)));
  JobImporter imp(job_wrapper, &context, module.get());
  if (succeeded(imp.ProcessJob())) {
    mlir::PassManager pm(&context);
    pm.addPass(createCanonicalizerPass());
    if (mlir::failed(pm.run(*module))) {
      module->emitError("Failed to run canonicalizer pass");
      exit(EXIT_FAILURE);
    }

    std::string mlir;
    llvm::raw_string_ostream os_mlir(mlir);
    module->print(os_mlir);
    return mlir;
  } else {
    const auto& job_name = job->job_conf().job_name();
    llvm::errs() << "Failed to convert Job to IR, job_name: " << job_name << "\n";
    exit(EXIT_FAILURE);
  }
}

void SaveJobToIR(RoundTripOneFlowJobWrapperInterface& job_wrapper, const std::string& path) {
  const ::oneflow::Job* job = job_wrapper.job();
  mlir::MLIRContext context;
  context.getOrLoadDialect<oneflow::OneFlowDialect>();
  context.loadDialect<mlir::func::FuncDialect>();

  OwningOpRef<ModuleOp> module(
      ModuleOp::create(FileLineColLoc::get(&context, "", /*line=*/0, /*column=*/0)));
  JobImporter imp(job_wrapper, &context, module.get());
  if (succeeded(imp.ProcessJob())) {
    mlir::PassManager pm(&context);
    pm.addPass(createCanonicalizerPass());
    if (mlir::failed(pm.run(*module))) {
      module->emitError("Failed to run canonicalizer pass");
      exit(EXIT_FAILURE);
    }

    std::string mlir;
    llvm::raw_string_ostream os_mlir(mlir);
    module->print(os_mlir);
    std::string filename = path + "/model.mlir";
    std::ofstream fs(filename, std::ios::trunc);
    if (!fs.is_open()) {
      llvm::errs() << "fail to open file " << filename;
      exit(EXIT_FAILURE);
    }
    fs << mlir;
    fs.close();
  } else {
    const auto& job_name = job->job_conf().job_name();
    llvm::errs() << "fail to convert job to IR, job_name: " << job_name << "\n";
    exit(EXIT_FAILURE);
  }
}

void LoadJobFromIR(RoundTripOneFlowJobWrapperInterface& job_wrapper, const std::string& path) {
  MLIRContext context;
  context.getOrLoadDialect<oneflow::OneFlowDialect>();
  context.loadDialect<mlir::func::FuncDialect>();
  OwningOpRef<ModuleOp> module = parseSourceFile<ModuleOp>(path, &context);
  if (!module) {
    llvm::errs() << "fail to parse file: " << path << "\n";
    exit(EXIT_FAILURE);
  }
  JobImporter imp(job_wrapper, &context, module.get());
  if (failed(imp.TryToUpdateJob())) {
    llvm::errs() << "fail to load job from IR";
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
