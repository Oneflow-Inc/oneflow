#ifndef ONEFLOW_CORE_GRAPH_TASK_GRAPH_H_
#define ONEFLOW_CORE_GRAPH_TASK_GRAPH_H_

#include "oneflow/core/graph/logical_graph.h"
#include "oneflow/core/job/id_manager.h"
#include "oneflow/core/job/parallel_desc.h"
#include "oneflow/core/operator/operator.h"

namespace oneflow {

class TaskGraph final : public Graph<TaskNode, TaskEdge> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(TaskGraph);
  TaskGraph() = delete;
  ~TaskGraph() = default;

  TaskGraph(std::unique_ptr<const LogicalGraph>&& logical_gph);

  const char* TypeName() const override { return "TaskGraph"; }

#define DECLARE_BLD_SUB_TASK_GRAPH_METHOD(method_name) void method_name BLD_SUB_TSK_GPH_MTHD_ARGS();

  DECLARE_BLD_SUB_TASK_GRAPH_METHOD(BldSubTskGphByBoxing);
  DECLARE_BLD_SUB_TASK_GRAPH_METHOD(BldSubTskGphByOneToOne);
  DECLARE_BLD_SUB_TASK_GRAPH_METHOD(BldSubTskGphBySelectOneSourceToSoleSink);
  DECLARE_BLD_SUB_TASK_GRAPH_METHOD(BldSubTskGphByReduceScatter2ReduceAdd);
  DECLARE_BLD_SUB_TASK_GRAPH_METHOD(BldSubTskGphByReduceAdd2ReduceGather);

 private:
  TaskNode* Build121BufTo(
      TaskNode* src, int64_t dst_machine_id, int32_t dst_mem_zone_id,
      std::function<TaskNode*(int64_t machine_id, int32_t mem_zone_id)> Get121BufTask,
      std::function<TaskNode*(int64_t machine_id, int32_t mem_zone_id, TaskNode*)> Set121BufTask);
  TaskNode* AddCopyH2DTaskIfNotCpu(TaskNode*);
  TaskNode* AddCopyD2HTaskIfNotCpu(TaskNode*);
  void AddCopyCommNetTask(TaskNode* src, TaskNode* dst);
  void BuildOutBoxing(const LogicalNode* logical,
                      const std::vector<CompTaskNode*>& sorted_comp_tasks,
                      std::vector<TaskNode*>* sorted_out_box,
                      std::function<int64_t(const TaskNode*)> AllocateCpuThrdId);
  void BuildInBoxing(const LogicalNode* logical,
                     const std::vector<CompTaskNode*>& sorted_comp_tasks,
                     std::vector<TaskNode*>* sorted_in_box,
                     std::function<int64_t(const TaskNode*)> AllocateCpuThrdId);
  void ConnectWithCopyCommNetIfNeed(TaskNode* src, TaskNode* dst);

  std::unique_ptr<const LogicalGraph> logical_gph_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_GRAPH_TASK_GRAPH_H_
