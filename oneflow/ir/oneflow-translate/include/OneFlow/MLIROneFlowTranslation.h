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
#ifndef ONEFLOW_IR_ONEFLOW_TRANSLATE_INCLUDE_ONEFLOW_MLIRONEFLOWTRANSLATION_H_
#define ONEFLOW_IR_ONEFLOW_TRANSLATE_INCLUDE_ONEFLOW_MLIRONEFLOWTRANSLATION_H_

#include "oneflow/core/framework/user_op_def.pb.h"
#include "oneflow/core/job/job.pb.h"
#include "oneflow/core/job/sbp_parallel.pb.h"
#include "oneflow/core/operator/op_conf.pb.h"
#include "OneFlow/SBP/SBPImporter.h"

#include "OneFlow/OneFlowOps.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"

#include <functional>
#include <string>

using UserOpArgs = const ::google::protobuf::Map<std::string, ::oneflow::UserOpConf_ListString>&;
using UserOpArgDefs = const ::google::protobuf::RepeatedPtrField<::oneflow::UserOpDef_ArgDef>&;

namespace mlir {

namespace oneflow {

// TODO: wrap in a helper namespace

LogicalResult IsAttrBelong2Op(const std::string& op_type_name, const std::string& attr_name);

LogicalResult ConvertUserOpInputs(Operation* op, StringRef op_name,
                                  ::oneflow::UserOpConf* user_conf);
LogicalResult ConvertUserOpOutputs(Operation* op, StringRef op_name,
                                   ::oneflow::UserOpConf* user_conf);
LogicalResult ConvertCtrlInputs(Operation* op, ::oneflow::OperatorConf& op_conf);
llvm::Optional<mlir::oneflow::DataTypeAttr> GetDataTypeAttr(MLIRContext* context,
                                                            ::oneflow::DataType oneflow_value);
LogicalResult ConvertVariableOpConf(VariableOp op, ::oneflow::OperatorConf* op_conf);
LogicalResult ConvertInputOpConf(InputOp op, ::oneflow::OperatorConf* op_conf);
LogicalResult ConvertOutputOpConf(OutputOp op, ::oneflow::OperatorConf* op_conf);

LogicalResult ParseNdSbpFromAttr(ArrayAttr nd_sbp_attr, ::oneflow::NdSbp* nd_sbp);
Attribute ConvertNdSbpToAttr(Builder& builder, const ::oneflow::NdSbp& nd_sbp);

class Importer {
 public:
  Importer(MLIRContext* context, ModuleOp module)
      : builder_(context),
        context_(context),
        module_(module),
        unknown_loc_(FileLineColLoc::get(context, "unknown_loc", 0, 0)) {}
  virtual ~Importer() = default;
  LogicalResult namedAttributesFromUserOp(const ::oneflow::OperatorConf& op,
                                          std::vector<NamedAttribute>& attr_vec);
  virtual LogicalResult AppendDataInOperand(const std::string& lbn,
                                            std::vector<::mlir::Value>& operand_vec) {
    return failure();
  }
  virtual LogicalResult AppendDataInOperand(const std::string& key, const int32_t index,
                                            const std::string& lbn,
                                            std::vector<::mlir::Value>& operand_vec) {
    return AppendDataInOperand(lbn, operand_vec);
  }
  virtual LogicalResult AppendCtrlInOperand(const ::oneflow::OperatorConf& op,
                                            std::vector<::mlir::Value>& operand_vec) = 0;
  LogicalResult AppendCtrlOutType(llvm::SmallVector<Type, 8>& out_types);
  LogicalResult AddOpConf(const ::oneflow::OperatorConf& op, std::vector<NamedAttribute>& attr_vec);
  LogicalResult AddUserOpInputOutputSegments(const ::oneflow::OperatorConf& op,
                                             std::vector<NamedAttribute>& attr_vec);
  virtual LogicalResult AddDeviceName(const ::oneflow::OperatorConf& op,
                                      std::vector<NamedAttribute>& attr_vec) = 0;
  LogicalResult AddOperandSegmentSizes(int32_t input_lbns_size, int32_t ctrl_in_size,
                                       std::vector<NamedAttribute>& attr_vec);
  LogicalResult AddResultSegmentSizes(int32_t output_lbns_size,
                                      std::vector<NamedAttribute>& attr_vec);
  virtual LogicalResult InsertOpResults(const ::oneflow::OperatorConf& op, Operation*) = 0;
  LogicalResult ProcessUserOp(const ::oneflow::OperatorConf& op);
  virtual LogicalResult ProcessSystemOp(const ::oneflow::OperatorConf& op) = 0;

  IntegerAttr getSI64IntegerAttr(int64_t value) {
    return IntegerAttr::get(GetBuilder().getIntegerType(64, /*isSigned=*/true),
                            APInt(64, value, /*isSigned=*/true));
  }
  ArrayAttr getSI32ArrayAttr(ArrayRef<int32_t> values) {
    auto attrs = llvm::to_vector<8>(llvm::map_range(
        values, [this](int32_t v) -> Attribute { return GetBuilder().getSI32IntegerAttr(v); }));
    return GetBuilder().getArrayAttr(attrs);
  }
  ArrayAttr getSI64ArrayAttr(ArrayRef<int64_t> values) {
    auto attrs = llvm::to_vector<8>(
        llvm::map_range(values, [this](int64_t v) -> Attribute { return getSI64IntegerAttr(v); }));
    return GetBuilder().getArrayAttr(attrs);
  }

  ArrayAttr GetAttrFromShape(const ::oneflow::ShapeProto& shape);
  ArrayAttr GetAttrFromStride(const ::oneflow::Int64ListProto& stride);
  OpBuilder& GetBuilder() { return builder_; }
  MLIRContext* GetMLIRContext() { return context_; }
  ModuleOp& GetModule() { return module_; }
  Location& GetRootLocation() { return unknown_loc_; }
  virtual Type GetTensorTypeOfLbn(const std::string& lbn) = 0;
  void SetOpStateLoc(const ::oneflow::OperatorConf&, OperationState&);

 private:
  OpBuilder builder_;
  MLIRContext* context_;
  ModuleOp module_;
  Location unknown_loc_;
};

class RoundTripOneFlowJobWrapperInterface {
 public:
  virtual ~RoundTripOneFlowJobWrapperInterface() {}
  virtual const ::oneflow::Job* job() const = 0;
  virtual void UpdateJob(::oneflow::Job* new_job) = 0;
  virtual void DumpLog(const std::string& filename, const std::string& content) = 0;
  virtual const ::oneflow::ParallelConf& ParallelConf4OpName(const std::string& op_name) const = 0;
  virtual const ::oneflow::OperatorConf& OpConf4OpName(const std::string& op_name) const = 0;
  virtual std::pair<std::vector<std::string>, std::vector<std::string>> InputBns4OpName(
      const std::string& op_name) const = 0;
  virtual std::vector<std::string> OutputLbns4OpName(const std::string& op_name) const = 0;
  virtual std::string ReplaceInputLbnInOpCustomizedConf(::oneflow::OperatorConf* op_conf,
                                                        const std::string& ibn,
                                                        const std::string& new_val) const = 0;
  virtual void QueryLogicalBlob(
      const std::string& lbn, std::function<void(const int64_t* shape_begin,
                                                 const int64_t* shape_end, ::oneflow::DataType dt)>
                                  cb) const = 0;
  virtual void TopoForEachOpConf(
      std::function<void(const ::oneflow::OperatorConf*)> Handler) const = 0;
  virtual bool IsLastIRPass() const = 0;
};

void RoundTripOneFlowJob(
    RoundTripOneFlowJobWrapperInterface& job_wrapper,
    const std::function<bool(::oneflow::Job* job, std::string& reason)>& is_legit_job);

void registerFromOneFlowJobTranslation();

std::string ConvertJobToTosaIR(RoundTripOneFlowJobWrapperInterface& job_wrapper);
void SaveJobToIR(RoundTripOneFlowJobWrapperInterface& job_wrapper, const std::string& path);
std::string ConvertJobToIR(RoundTripOneFlowJobWrapperInterface& job_wrapper);
void LoadJobFromIR(RoundTripOneFlowJobWrapperInterface& job_wrapper, const std::string& path);

}  // namespace oneflow

}  // namespace mlir

#endif  // ONEFLOW_IR_ONEFLOW_TRANSLATE_INCLUDE_ONEFLOW_MLIRONEFLOWTRANSLATION_H_
