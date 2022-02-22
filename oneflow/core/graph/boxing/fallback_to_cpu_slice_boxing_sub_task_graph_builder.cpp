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
#include "oneflow/core/graph/boxing/fallback_to_cpu_slice_boxing_sub_task_graph_builder.h"
#include "oneflow/core/graph/boxing/sub_task_graph_builder_util.h"

namespace oneflow {

Maybe<SubTskGphBuilderStatus> FallbackToCpuSliceBoxingSubTskGphBuilder::Build(
    SubTskGphBuilderCtx* ctx, const std::vector<TaskNode*>& sorted_in_tasks,
    std::vector<TaskNode*>* sorted_out_tasks,
    std::vector<std::vector<TaskNode*>>* sorted_ctrl_tasks, const ParallelDesc& in_parallel_desc,
    const ParallelDesc& out_parallel_desc, const LogicalBlobId& lbi,
    const BlobDesc& logical_blob_desc, const SbpParallel& in_sbp_parallel,
    const SbpParallel& out_sbp_parallel, const Shape& time_shape) const {
  std::vector<SubTskGphBuilderStatus> status;

  std::vector<TaskNode*> cpu_in_tasks;
  std::vector<TaskNode*> cpu_out_tasks;
  std::vector<std::vector<TaskNode*>> cpu_ctrl_tasks;
  cpu_out_tasks.reserve(out_parallel_desc.parallel_num());

  FOR_RANGE(int64_t, in_id, 0, in_parallel_desc.parallel_num()) {
    TaskNode* in_node = sorted_in_tasks.at(in_id);
    TaskNode* proxy_on_src_host = ctx->task_graph()->GetProxyNode(
        in_node, lbi, GetNodeCPUMemZoneId(in_node->MemZoneId121().rank()));
    cpu_in_tasks.push_back(proxy_on_src_host);
  }
  status.emplace_back("MoveToCpu", "-");

  ParallelConf cpu_in_parallel_conf = in_parallel_desc.parallel_conf();
  cpu_in_parallel_conf.set_device_tag("cpu");
  ParallelConf cpu_out_parallel_conf = out_parallel_desc.parallel_conf();
  cpu_out_parallel_conf.set_device_tag("cpu");
  Maybe<SubTskGphBuilderStatus> boxing_builder_status =
      TRY(builder_->Build(ctx, cpu_in_tasks, &cpu_out_tasks, &cpu_ctrl_tasks,
                          ParallelDesc(cpu_in_parallel_conf), ParallelDesc(cpu_out_parallel_conf),
                          lbi, logical_blob_desc, in_sbp_parallel, out_sbp_parallel, time_shape));
  if (!boxing_builder_status.IsOk()
      && SubTskGphBuilderUtil::IsErrorBoxingNotSupported(*boxing_builder_status.error())) {
    return Error::BoxingNotSupportedError();
  }
  status.push_back(*JUST(boxing_builder_status));

  FOR_RANGE(int64_t, out_id, 0, out_parallel_desc.parallel_num()) {
    TaskNode* out_node =
        ctx->task_graph()->GetProxyNode(cpu_out_tasks.at(out_id), lbi, out_parallel_desc, out_id);
    sorted_out_tasks->push_back(out_node);
  }
  status.emplace_back("MoveBackToDevice", "-");
  if (!cpu_ctrl_tasks.empty()) {
    CHECK_EQ(cpu_ctrl_tasks.size(), sorted_out_tasks->size());
    FOR_RANGE(size_t, i, 0, sorted_out_tasks->size()) {
      for (TaskNode* ctrl_node : cpu_ctrl_tasks.at(i)) {
        std::string regst_desc_name;
        ctrl_node->BuildCtrlRegstDesc(sorted_out_tasks->at(i), &regst_desc_name);
        TaskEdge* edge = ctx->task_graph()->NewEdge();
        Connect<TaskNode>(ctrl_node, edge, sorted_out_tasks->at(i));
        ctrl_node->BindEdgeWithProducedRegst(edge, regst_desc_name);
      }
    }
  }

  return TRY(MakeComposedSubTskGphBuilderStatus(status));
}

}  // namespace oneflow
