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
#ifndef ONEFLOW_CORE_JOB_REWRITER_AUTOGRAD_H_
#define ONEFLOW_CORE_JOB_REWRITER_AUTOGRAD_H_

#include "oneflow/core/job/job_desc.h"
#include "oneflow/core/operator/operator.h"
#include "oneflow/core/graph/op_graph.h"

namespace oneflow {

Maybe<void> MakePredicatorNeedBackwardOp(const OpGraph& op_graph,
                                         std::function<bool(OpNode*)>* NeedBackwardOp);

Maybe<void> AutoGrad(const OpGraph& op_graph, JobBuilder* job_builder,
                     HashMap<LogicalBlobId, LogicalBlobId>* out_lbi2out_diff_lbi);
void AddDiffParallelCast(const OpGraph& op_graph, JobBuilder* job_builder,
                         HashMap<LogicalBlobId, LogicalBlobId>* lbi2diff_lbi);
void AddDiffStaticShapeCast(const OpGraph& op_graph, JobBuilder* job_builder,
                            HashMap<LogicalBlobId, LogicalBlobId>* lbi2diff_lbi);
Maybe<void> ScaleModelDiffByLossInstanceNum(const OpGraph& op_graph, JobBuilder* job_builder,
                                            HashMap<LogicalBlobId, LogicalBlobId>* lbi2diff_lbi);
void ScaleModelDiffByLossScale(const OpGraph& op_graph, JobBuilder* job_builder,
                               HashMap<LogicalBlobId, LogicalBlobId>* lbi2diff_lbi);
void RegularizeGradient(const OpGraph& op_graph, JobBuilder* job_builder,
                        HashMap<LogicalBlobId, LogicalBlobId>* lbi2diff_lbi);
void ClipGradient(const OpGraph& op_graph, JobBuilder* job_builder,
                  HashMap<LogicalBlobId, LogicalBlobId>* lbi2diff_lbi, const ClipConf& clip_conf);
Maybe<void> GenerateBackwardOpConfIf(
    const Operator& op, std::vector<OperatorConf>* op_confs,
    const std::function<LogicalBlobId*(const std::string&)>& DiffLbi4BnInOp,
    const std::function<const BlobDesc&(const std::string&)>& LogicalBlobDesc4BnInOp);
void GetVariableOpNodesAndDescendants(const OpGraph& op_graph, HashSet<OpNode*>* op_nodes);

class GenerateBackwardOpConfWrapperStruct final {
 public:
  using NaiveFunc = std::function<void(const Operator&, std::vector<OperatorConf>*,
                                       const std::function<LogicalBlobId*(const std::string&)>&)>;
  using MaybeFunc =
      std::function<Maybe<void>(const Operator&, std::vector<OperatorConf>*,
                                const std::function<LogicalBlobId*(const std::string&)>&,
                                const std::function<const BlobDesc&(const std::string&)>&)>;
  GenerateBackwardOpConfWrapperStruct(const NaiveFunc& f)
      : naive_func_(std::make_unique<NaiveFunc>(f)) {}
  GenerateBackwardOpConfWrapperStruct(const MaybeFunc& f)
      : maybe_func_(std::make_unique<MaybeFunc>(f)) {}
  Maybe<void> Call(const Operator&, std::vector<OperatorConf>*,
                   const std::function<LogicalBlobId*(const std::string&)>&,
                   const std::function<const BlobDesc&(const std::string&)>&) const;

 private:
  const std::unique_ptr<const NaiveFunc> naive_func_;
  const std::unique_ptr<const MaybeFunc> maybe_func_;
};

#define REGISTER_OP_GRAD(op_type_case, gen_grad_func)                                \
  REGISTER_CLASS_CREATOR(int32_t, op_type_case, GenerateBackwardOpConfWrapperStruct, \
                         ([] { return new GenerateBackwardOpConfWrapperStruct(gen_grad_func); }))

}  // namespace oneflow

#endif  // ONEFLOW_CORE_JOB_REWRITER_AUTOGRAD_H_
