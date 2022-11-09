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

class JobPassCtx;

void AddDiffHalf2FloatCast(const OpGraph& op_graph, JobBuilder* job_builder,
                           HashMap<LogicalBlobId, LogicalBlobId>* lbi2diff_lbi);
void AddDiffParallelCast(const OpGraph& op_graph, JobBuilder* job_builder,
                         HashMap<LogicalBlobId, LogicalBlobId>* lbi2diff_lbi);
void AddDiffStaticShapeCast(const OpGraph& op_graph, JobBuilder* job_builder,
                            HashMap<LogicalBlobId, LogicalBlobId>* lbi2diff_lbi);
Maybe<void> CountNotFiniteIfNeeded(JobPassCtx* ctx, const OpGraph& op_graph,
                                   JobBuilder* job_builder,
                                   const HashMap<LogicalBlobId, LogicalBlobId>& lbi2diff_lbi);
Maybe<void> MakeGetterLossOpNode4OpName(
    const OpGraph& op_graph, std::function<OpNode*(const std::string&)>* LossOpNode4OpName);
Maybe<void> ScaleModelDiffByLossInstanceNum(const OpGraph& op_graph, JobBuilder* job_builder,
                                            HashMap<LogicalBlobId, LogicalBlobId>* lbi2diff_lbi);

Maybe<void> ScaleInitialDiffByLossScale(
    JobPassCtx* ctx, const OpGraph& op_graph, JobBuilder* job_builder,
    HashMap<LogicalBlobId, LogicalBlobId>* loss_lbi2initial_diff_lbi);

void ScaleModelDiffByLossScale(JobPassCtx* ctx, const OpGraph& op_graph, JobBuilder* job_builder,
                               HashMap<LogicalBlobId, LogicalBlobId>* lbi2diff_lbi);
void RegularizeGradient(const OpGraph& op_graph, JobBuilder* job_builder,
                        HashMap<LogicalBlobId, LogicalBlobId>* lbi2diff_lbi);
void ClipGradient(JobPassCtx* ctx, const OpGraph& op_graph, JobBuilder* job_builder,
                  HashMap<LogicalBlobId, LogicalBlobId>* lbi2diff_lbi, const ClipConf& clip_conf);
}  // namespace oneflow

#endif  // ONEFLOW_CORE_JOB_REWRITER_AUTOGRAD_H_
