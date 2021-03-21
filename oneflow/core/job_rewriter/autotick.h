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
#ifndef ONEFLOW_CORE_JOB_REWRITER_AUTOTICK_H_
#define ONEFLOW_CORE_JOB_REWRITER_AUTOTICK_H_

#include "oneflow/core/job/job_desc.h"
#include "oneflow/core/operator/operator.h"
#include "oneflow/core/graph/op_graph.h"

namespace oneflow {

void AutoPrependTick(const OpGraph& op_graph, JobBuilder* job_builder);
void AddTickForTimeShape(const OpGraph& op_graph, JobBuilder* job_builder);
void AutoSourceAndSinkTick(const OpGraph& op_graph, JobBuilder* job_builder);
void AddGlobalInputCriticalSections(const OpGraph& op_graph, JobBuilder* job_builder);
void AddGlobalOutputCriticalSections(const OpGraph& op_graph, JobBuilder* job_builder);

class MutOpConTickInputHelper {
 public:
  bool IsTickInputBound() const { return VirtualIsTickInputBound(); }
  virtual bool VirtualIsTickInputBound() const = 0;
  virtual OperatorConf NewTickInputBoundOpConf(const std::string& lbn) const = 0;
  void InitFromOpConf(const OperatorConf& op_conf) { op_conf_ = &op_conf; }

 protected:
  MutOpConTickInputHelper() : op_conf_(nullptr) {}
  const OperatorConf& op_conf() const { return *op_conf_; }

 private:
  const OperatorConf* op_conf_;
};

#define REGISTER_AUTO_TICK(op_type_case, HelperType) \
  REGISTER_CLASS(int32_t, op_type_case, MutOpConTickInputHelper, HelperType)

}  // namespace oneflow

#endif  // ONEFLOW_CORE_JOB_REWRITER_AUTOTICK_H_
