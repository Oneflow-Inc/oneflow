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
#ifndef ONEFLOW_MLIRONEFLOWTRANSLATION_H
#define ONEFLOW_MLIRONEFLOWTRANSLATION_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "oneflow/core/framework/user_op_def.pb.h"
#include "oneflow/core/job/job.pb.h"
#include "oneflow/core/operator/op_conf.pb.h"
#include <functional>
#include <string>
namespace mlir {

class Importer {
 public:
  Importer(MLIRContext* context, ModuleOp module)
      : builder_(context),
        context_(context),
        module_(module),
        unknown_loc_(FileLineColLoc::get(context, "unknown_loc", 0, 0)) {}

  OpBuilder& GetBuilder() { return builder_; }
  MLIRContext* GetMLIRContext() { return context_; }
  ModuleOp& GetModule() { return module_; }
  Location& GetRootLocation() { return unknown_loc_; }

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
  virtual ::oneflow::AttrType QueryAttrType(const std::string& op_type_name,
                                            const std::string& attr_name) const = 0;
  virtual void QueryLogicalBlob(
      const std::string& lbn, std::function<void(const int64_t* shape_begin,
                                                 const int64_t* shape_end, ::oneflow::DataType dt)>
                                  cb) const = 0;
  virtual void TopoForEachOpConf(
      std::function<void(const ::oneflow::OperatorConf*)> Handler) const = 0;
};

void RoundTripOneFlowJob(
    RoundTripOneFlowJobWrapperInterface& job_wrapper,
    const std::function<bool(::oneflow::Job* job, std::string& reason)>& is_legit_job);
void registerFromOneFlowJobTranslation();

}  // namespace mlir

#endif /* ONEFLOW_MLIRONEFLOWTRANSLATION_H */
