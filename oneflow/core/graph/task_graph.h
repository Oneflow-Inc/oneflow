#ifndef ONEFLOW_CORE_GRAPH_TASK_GRAPH_H_
#define ONEFLOW_CORE_GRAPH_TASK_GRAPH_H_

#include "oneflow/core/graph/logical_graph.h"
#include "oneflow/core/job/id_manager.h"
#include "oneflow/core/job/parallel_desc.h"
#include "oneflow/core/operator/operator.h"
#include "oneflow/core/graph/copy_task_node.h"

namespace oneflow {

class TaskGraph final : public Graph<TaskNode, TaskEdge> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(TaskGraph);
  TaskGraph() = delete;
  ~TaskGraph() = default;

  TaskGraph(std::unique_ptr<const LogicalGraph>&& logical_gph);

  const char* TypeName() const override { return "TaskGraph"; }
  void RemoveEmptyRegsts();
  void AddOrderingCtrlEdgeInSameChain();
  void AddMdUpdtCtrlEdgesWithinReduceSplitNode();
  void AddReduceNoBwForwardNodeOverlapingCtrlEdges();

  void EnableInplaceMemSharingInReduceStruct();
  void EnableInplaceMemSharing(const std::function<bool(const LogicalBlobId&, const std::string&)>&
                                   IsLbiAllConsumersReachableToOpName);

  void AddOrderCtrlEdgeBetweenCopyAndMdUpdt();
  void AcyclicTopoForEachNode(std::function<void(TaskNode* node)> Handler) const;
  void MdUpdtDelayedTopoForEachNode(std::function<void(TaskNode* node)> Handler) const;

#define DECLARE_BLD_SUB_TASK_GRAPH_METHOD(method_name) void method_name BLD_SUB_TSK_GPH_MTHD_ARGS();

  DECLARE_BLD_SUB_TASK_GRAPH_METHOD(BldSubTskGphByBoxing);
  DECLARE_BLD_SUB_TASK_GRAPH_METHOD(BldSubTskGphByBoxingV1);
  DECLARE_BLD_SUB_TASK_GRAPH_METHOD(BldSubTskGphByBoxingV2);
  DECLARE_BLD_SUB_TASK_GRAPH_METHOD(BldSubTskGphByOneToOne);
  DECLARE_BLD_SUB_TASK_GRAPH_METHOD(BldSubTskGphByRecordLoadToTick);
  DECLARE_BLD_SUB_TASK_GRAPH_METHOD(BldSubTskGphByBroadcastToBroadcast);
  DECLARE_BLD_SUB_TASK_GRAPH_METHOD(BldSubTskGphBySelectOneSourceToSoleSink);
  DECLARE_BLD_SUB_TASK_GRAPH_METHOD(BldSubTskGphByReduceScatter2ReduceAdd);
  DECLARE_BLD_SUB_TASK_GRAPH_METHOD(BldSubTskGphByReduceAdd2ReduceGather);
  DECLARE_BLD_SUB_TASK_GRAPH_METHOD(BldSubTskGphByReduceGather2ReduceGather);
  DECLARE_BLD_SUB_TASK_GRAPH_METHOD(BldSubTskGphByConnectNodeOnSameGpuDevice);
  DECLARE_BLD_SUB_TASK_GRAPH_METHOD(BldSubTskGphByPartialInLbiConnect);
  DECLARE_BLD_SUB_TASK_GRAPH_METHOD(BldSubTskGphByPartialOutLbiConnect);

 private:
  void AcyclicTopoForEachNode(std::function<bool(TaskNode* node)> IsAllowedStartNode,
                              std::function<void(TaskNode* node)> Handler) const;

  void BuildTaskPath(
      CompTaskNode* src, CompTaskNode* dst,
      std::function<TaskNode**(CompTaskNode* src, int64_t machine_id, int32_t mem_zone_id)>
          MutBufTask,
      bool use_buf_task_node);
  TaskNode* BuildTaskStep(
      TaskNode* cur_node, TaskNode* dst,
      std::function<TaskNode*(int64_t machine_id, int32_t mem_zone_id)> GetBufTask,
      std::function<TaskNode*(int64_t machine_id, int32_t mem_zone_id, TaskNode*)> SetBufTask,
      bool use_buf_task_node);
  TaskNode* TryAddCopyH2DTaskTo(TaskNode*);
  TaskNode* AddCopyD2HTaskFrom(TaskNode*);
  TaskNode* AddCopyCommNetTaskBetween(TaskNode* src, TaskNode* dst);
  void BuildOutBoxing(
      const LogicalNode* logical, const std::vector<CompTaskNode*>& sorted_comp_tasks,
      std::vector<TaskNode*>* sorted_out_box,
      std::function<TaskNode**(CompTaskNode* src, int64_t machine_id, int32_t mem_zone_id)>
          MutBufTask,
      std::function<int64_t(const TaskNode*)> AllocateCpuThrdId);
  void BuildInBoxing(const LogicalNode* logical,
                     const std::vector<CompTaskNode*>& sorted_comp_tasks,
                     std::vector<TaskNode*>* sorted_in_box,
                     std::function<int64_t(const TaskNode*)> AllocateCpuThrdId);
  void ConnectWithCopyCommNetIfNeed(TaskNode* src, TaskNode* dst);
  void ConnectCtrlEdges(const std::vector<CompTaskNode*>& src_task_nodes,
                        const std::vector<CompTaskNode*>& dst_task_nodes, int64_t ctrl_regst_num);

  void SetAreaIdForNewNodes(const LogicalNode* src_logical, const LogicalNode* dst_logical);
  void MergeChainAndSetOrderInGraphForEachNode();
  void BuildCtrlRegstDescInSameChain();

  void GenerateIndependentThrdId(
      const std::vector<std::pair<int64_t, CompTaskNode*>>& persistence_nodes);

  // inplace
  void GetInplaceOpBlobArgList(
      OpBlobArgList* inplace_obas, const HashSet<TaskNode*>& dev_nodes,
      const std::function<const TaskNode*(const std::string&)>& TaskNode4OpName) const;
  void GetSafeInplaceOpBlobArgList(
      OpBlobArgList* obas, const HashSet<TaskNode*>& dev_nodes,
      const std::function<bool(const LogicalBlobId&, const std::string&)>&
          IsLbiConsumersReachableToOpName) const;
  void SetTaskRegstInplaceInfo(const OpBlobArgList& obas,
                               const HashSet<TaskNode*>& dev_nodes) const;
  void ForEachGpuDeviceNodes(
      const std::function<void(const HashSet<TaskNode*>& dev_nodes)>& Handler) const;

  std::unique_ptr<const LogicalGraph> logical_gph_;
  std::vector<TaskNode*> ordered_task_nodes_;
};

bool IsBackEdge(TaskNode* src, TaskNode* dst);

}  // namespace oneflow

#endif  // ONEFLOW_CORE_GRAPH_TASK_GRAPH_H_
