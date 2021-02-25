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
#ifndef ONEFLOW_CORE_GRAPH_TASK_GRAPH_H_
#define ONEFLOW_CORE_GRAPH_TASK_GRAPH_H_

#include "oneflow/core/graph/logical_graph.h"
#include "oneflow/core/job/id_manager.h"
#include "oneflow/core/job/parallel_desc.h"
#include "oneflow/core/operator/operator.h"
#include "oneflow/core/graph/copy_task_node.h"
#include "oneflow/core/register/op_blob_arg_info.h"
#include "oneflow/core/graph/boxing/boxing_logger.h"

namespace oneflow {

class SubTskGphBuilder;
class SubTskGphBuilderCtx;

class TaskGraph final : public Graph<TaskNode, TaskEdge> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(TaskGraph);
  TaskGraph() = delete;
  ~TaskGraph() override = default;

  explicit TaskGraph(std::unique_ptr<const LogicalGraph>&& logical_gph);

  const char* TypeName() const override { return "TaskGraph"; }
  void RemoveEmptyRegsts();
  void MergeChainAndAddOrderingCtrlEdgeInSameChain();

  void EnableInplaceMemSharing(const std::function<bool(const std::string&, const std::string&)>&
                                   IsOpNameDataOrCtrlReachable);

  void AcyclicTopoForEachNode(const std::function<void(TaskNode* node)>& Handler) const;

#define DECLARE_BLD_SUB_TASK_GRAPH_METHOD(method_name) void method_name BLD_SUB_TSK_GPH_MTHD_ARGS();

  DECLARE_BLD_SUB_TASK_GRAPH_METHOD(BldSubTskGphByBoxing);
  DECLARE_BLD_SUB_TASK_GRAPH_METHOD(BldSubTskGphByOneToOne);
  DECLARE_BLD_SUB_TASK_GRAPH_METHOD(BldSubTskGphByBroadcastToBroadcast);
  DECLARE_BLD_SUB_TASK_GRAPH_METHOD(BldSubTskGphByPartialInLbiConnect);
  DECLARE_BLD_SUB_TASK_GRAPH_METHOD(BldSubTskGphByPartialOutLbiConnect);
  DECLARE_BLD_SUB_TASK_GRAPH_METHOD(BldSubTskGphBySrcSubsetConnect);
  DECLARE_BLD_SUB_TASK_GRAPH_METHOD(BldSubTskGphByDstSubsetConnect);
  DECLARE_BLD_SUB_TASK_GRAPH_METHOD(BldSubTskGphNormalForwardToDecodeH2D);

 private:
  void AcyclicTopoForEachNode(std::function<bool(TaskNode* node)> IsAllowedStartNode,
                              const std::function<void(TaskNode* node)>& Handler) const;

  void BuildTaskPath(CompTaskNode* src, CompTaskNode* dst, MutBufTaskFn MutBufTask,
                     bool use_buf_task_node);
  using GetBufTaskFn = std::function<TaskNode*(int64_t, MemZoneId)>;
  using SetBufTaskFn = std::function<TaskNode*(int64_t, MemZoneId, TaskNode*)>;
  TaskNode* BuildTaskStep(TaskNode* src, TaskNode* dst, const GetBufTaskFn& GetBufTask,
                          const SetBufTaskFn& SetBufTask, bool use_buf_task_node);

  TaskNode* TryAddCopyH2DTaskTo(TaskNode*);
  TaskNode* AddCopyD2HTaskFrom(TaskNode*);
  TaskNode* AddCopyCommNetTaskBetween(TaskNode* src, TaskNode* dst);
  void ConnectWithCopyCommNetIfNeed(TaskNode* src, TaskNode* dst);
  Maybe<void> ConnectSrcSubsetTickEdges(const std::vector<CompTaskNode*>& src_task_nodes,
                                        const std::vector<CompTaskNode*>& dst_task_nodes);
  Maybe<void> ConnectDstSubsetTickEdges(const std::vector<CompTaskNode*>& src_task_nodes,
                                        const std::vector<CompTaskNode*>& dst_task_nodes);
  void ConnectCtrlEdges(const std::vector<CompTaskNode*>& src_task_nodes,
                        const std::vector<CompTaskNode*>& dst_task_nodes, int64_t ctrl_regst_num);

  void SetAreaIdForNewNodes(const LogicalNode* src_logical, const LogicalNode* dst_logical);
  void SetOrderInGraphForEachNode();
  void MergeChain();
  void BuildCtrlRegstDescInSameChain();

  // void GenerateIndependentThrdId(
  //     const std::vector<std::pair<int64_t, CompTaskNode*>>& persistence_nodes);

  // inplace
  void GetInplaceOpBlobArgList(
      InplaceObasInfo* obas_info, const HashSet<TaskNode*>& dev_nodes,
      const std::function<const TaskNode*(const std::string&)>& TaskNode4OpName) const;
  void GetSafeInplaceOpBlobArgList(
      InplaceObasInfo* safe_obas_info, const HashSet<TaskNode*>& dev_nodes,
      const std::function<bool(const std::string&, const std::string&)>&
          IsOpNameDataOrCtrlReachable) const;
  void SetTaskRegstInplaceInfo(const InplaceObasInfo& obas_info,
                               const HashSet<TaskNode*>& dev_nodes) const;
  void ForEachGpuDeviceNodes(
      const std::function<void(const HashSet<TaskNode*>& dev_nodes)>& Handler) const;

  std::unique_ptr<const LogicalGraph> logical_gph_;
  std::vector<TaskNode*> ordered_task_nodes_;
  std::shared_ptr<SubTskGphBuilder> sub_tsk_gph_builder_;
  std::shared_ptr<SubTskGphBuilderCtx> sub_tsk_gph_builder_ctx_;
  std::unique_ptr<BoxingLogger> boxing_logger_;
};

bool IsBackEdge(TaskNode* src, TaskNode* dst);

}  // namespace oneflow

#endif  // ONEFLOW_CORE_GRAPH_TASK_GRAPH_H_
