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
#ifndef ONEFLOW_CORE_GRAPH_BOXING_HIERARCHICAL_SUB_TASK_GRAPH_BUILDER_IMPL_H_
#define ONEFLOW_CORE_GRAPH_BOXING_HIERARCHICAL_SUB_TASK_GRAPH_BUILDER_IMPL_H_

#include "oneflow/core/graph/boxing/hierarchical_sub_task_graph_builder.h"

namespace oneflow {

class DispatchHierarchicalSubTskGphBuilder final : public HierarchicalSubTskGphBuilder {
 public:
  OF_DISALLOW_COPY_AND_MOVE(DispatchHierarchicalSubTskGphBuilder);
  DispatchHierarchicalSubTskGphBuilder();
  ~DispatchHierarchicalSubTskGphBuilder() override;

  Maybe<SubTskGphBuilderStatus> Build(SubTskGphBuilderCtx* ctx,
                                      const std::vector<TaskNode*>& sorted_in_tasks,
                                      std::vector<TaskNode*>* sorted_out_tasks,
                                      std::vector<std::vector<TaskNode*>>* sorted_ctrl_tasks,
                                      const ParallelDesc& in_parallel_desc,
                                      const ParallelDesc& out_parallel_desc,
                                      const LogicalBlobId& lbi, const BlobDesc& logical_blob_desc,
                                      const ParallelDistribution& in_parallel_distribution,
                                      const ParallelDistribution& out_parallel_distribution,
                                      const Shape& time_shape) const override;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

void InOutParallelDimReduce(const ParallelDesc& in_parallel_desc,
                            const ParallelDesc& out_parallel_desc,
                            const ParallelDistribution& in_parallel_distribution,
                            const ParallelDistribution& out_parallel_distribution,
                            ParallelDesc* reduced_in_parallel_desc,
                            ParallelDesc* reduced_out_parallel_desc,
                            ParallelDistribution* reduced_in_parallel_distribution,
                            ParallelDistribution* reduced_out_parallel_distribution);

}  // namespace oneflow

#endif  // ONEFLOW_CORE_GRAPH_BOXING_HIERARCHICAL_SUB_TASK_GRAPH_BUILDER_IMPL_H_
