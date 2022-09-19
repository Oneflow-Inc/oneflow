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

#include "oneflow/core/job/id_manager.h"
#include "oneflow/core/job/parallel_desc.h"
#include "oneflow/core/operator/operator.h"
#include "oneflow/core/graph/op_graph.h"
#include "oneflow/core/graph/compute_task_node.h"
#include "oneflow/core/graph/copy_task_node.h"
#include "oneflow/core/register/op_blob_arg_info.h"
#include "oneflow/core/graph/boxing/boxing_logger.h"
#include "oneflow/core/memory/memory_zone.h"

namespace oneflow {

class SubTskGphBuilderCtx;
class HierarchicalSubTskGphBuilder;

#define BLD_SUB_TSK_GPH_MTHD_ARGS()                                                \
  (const OpEdge* op_edge, const std::vector<CompTaskNode*>& sorted_src_comp_tasks, \
   const std::vector<CompTaskNode*>& sorted_dst_comp_tasks)

class TaskGraph;
using BldSubTskGphMthd = void(TaskGraph::*) BLD_SUB_TSK_GPH_MTHD_ARGS();

class TaskGraph : public Graph<TaskNode, TaskEdge> {
 proteced:
  explicit TaskGraph(bool enable_straighten_algorithm);

 public:
  OF_DISALLOW_COPY_AND_MOVE(TaskGraph);
  virtual ~TaskGraph() override;

  const char* TypeName() const override { return "TaskGraph"; }
  void RemoveEmptyRegsts();
  void MergeChainAndAddOrderingCtrlEdgeInSameChain();

  void EnableInplaceMemSharing(const std::function<bool(const std::string&, const std::string&)>&
                                   IsOpNameDataOrCtrlReachable);

  TaskNode* GetProxyNode(TaskNode* src_node, const LogicalBlobId& lbi,
                         const MemZoneId& dst_mem_zone_id);

  TaskNode* GetProxyNode(TaskNode* src_node, const LogicalBlobId& lbi,
                         const ParallelDesc& dst_parallel_desc, int64_t dst_parallel_id);

  TaskEdge* NewTaskEdgeWithLbi(const LogicalBlobId& lbi);
  TaskEdge* NewTaskEdgeWithLbis(const std::vector<LogicalBlobId>& lbis);

  void ConnectWithLbi(TaskNode* src_node, TaskNode* dst_node, const LogicalBlobId& lbi);

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
  void BuildTaskPath(TaskNode* src_node, TaskNode* dst_node, const LogicalBlobId& lbi);

  void ConnectCtrlEdges(const std::vector<CompTaskNode*>& src_task_nodes,
                        const std::vector<CompTaskNode*>& dst_task_nodes);

  void ConnectCtrlEdge(const CompTaskNode* src_task_node, const CompTaskNode* dst_task_node);

  void SetOrderInGraphForEachNode();
  void MergeChain();
  void BuildCtrlRegstDescInSameChain();

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

  std::vector<TaskNode*> ordered_task_nodes_;
  std::unique_ptr<HierarchicalSubTskGphBuilder> hierarchical_sub_tsk_gph_builder_;
  std::unique_ptr<SubTskGphBuilderCtx> sub_tsk_gph_builder_ctx_;
  std::unique_ptr<BoxingLogger> boxing_logger_;

  struct ProxyKey {
    TaskNode* src_node;
    LogicalBlobId lbi;
    MemZoneId dst_mem_zone_id;

    ProxyKey(TaskNode* src, const LogicalBlobId& arg_lbi, const MemZoneId& arg_mem_zone_id)
        : src_node(src), lbi(arg_lbi), dst_mem_zone_id(arg_mem_zone_id) {}

    bool operator==(const ProxyKey& other) const {
      return src_node == other.src_node && lbi == other.lbi
             && dst_mem_zone_id == other.dst_mem_zone_id;
    }

    struct Hasher {
      inline size_t operator()(const ProxyKey& key) const {
        return Hash(key.src_node, key.lbi, key.dst_mem_zone_id.hash());
      }
    };
  };

  HashMap<ProxyKey, TaskNode*, ProxyKey::Hasher> proxy2node;
};

class GlobalTaskGraph final : public TaskGraph {
 public:
  OF_DISALLOW_COPY_AND_MOVE(GlobalTaskGraph);
  explicit GlobalTaskGraph(bool enable_straighten_algorithm)
    : TaskGraph(enable_straighten_algorithm) {}
};

class BoxingTaskGraphProto;

class BoxingTaskGraph final : public TaskGraph {
 public:
  OF_DISALLOW_COPY_AND_MOVE(BoxingTaskGraph);
  ~BoxingTaskGraph() = default;

  static Maybe<BoxingTaskGraph> New() {
    std::shared_ptr<BoxingTaskGraph> graph(new BoxingTaskGraph());
    JUST(graph->Init());
    return graph;
  }

  void ToProto(BoxingTaskGraphProto* proto) const;

 private:
  BoxingTaskGraph() = default;
  Maybe<void> Init();

  HashMap<const OpNode*, std::vector<CompTaskNode*>> op_node2sorted_comp_tasks_;
};

class TaskGraphRebuildCtx;

class RankTaskGraph final : public TaskGraph {
 public:
  OF_DISALLOW_COPY_AND_MOVE(RankTaskGraph);
  ~RankTaskGraph() = default;

  static Maybe<RankTaskGraph> New(const std::shared_ptr<BoxingTaskGraphProto>& boxing_task_graph_proto, int64_t rank, bool enable_straighten_algorithm) {
    std::shared_ptr<RankTaskGraph> graph(new RankTaskGraph(boxing_task_graph_proto, rank));
    JUST(graph->Init(enable_straighten_algorithm));
    return graph;
  }

 private:
  RankTaskGraph(const std::shared_ptr<BoxingTaskGraphProto>& boxing_task_graph_proto, int64_t rank)
      : boxing_task_graph_proto_(boxing_task_graph_proto), rank_(rank) {}

  Maybe<void> Init(bool enable_straighten_algorithm);
  Maybe<void> AddBoxingReletedCompTaskNodesFromProto();
  Maybe<void> CreateAndPartiallyInitTransportTaskNodesFromProto();
  Maybe<void> AddTransportTaskEdgesFromProto();
  Maybe<void> InitTransportTaskNodesFromProto();
  Maybe<TaskNode*> TryGetBoxingRelatedComTaskNode(const OpNode* op_node, int64_t parallel_id);
  Maybe<TaskNode*> TryCreateOrFindRankCompTaskNode(const OpNode* op_node);
  Maybe<TaskNode*> TryGetRankCompTaskNode(const OpNode* op_node);

  std::shared_ptr<BoxingTaskGraphProto> boxing_task_graph_proto_;
  const int64_t rank_;
  std::unique_ptr<TaskGraphRebuildCtx> task_graph_rebuild_ctx_;
  HashMap<const OpNode*, TaskNode*> op_node2comp_task_node_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_GRAPH_TASK_GRAPH_H_
