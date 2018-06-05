#ifndef ONEFLOW_CORE_GRAPH_MEMORY_SHARED_TASK_GRAPH_H_
#define ONEFLOW_CORE_GRAPH_MEMORY_SHARED_TASK_GRAPH_H_

#include "oneflow/core/graph/graph.h"
#include "oneflow/core/job/plan.pb.h"

namespace oneflow {

class MemSharedTaskNode;

class MemSharedTaskEdge final : public Edge<MemSharedTaskNode, MemSharedTaskEdge> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(MemSharedTaskEdge);
  MemSharedTaskEdge() = default;
  ~MemSharedTaskEdge() = default;
};

class MemSharedTaskNode final : public Node<MemSharedTaskNode, MemSharedTaskEdge> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(MemSharedTaskNode);
  explicit MemSharedTaskNode(const TaskProto& task_proto) : task_proto_(&task_proto) {}
  ~MemSharedTaskNode() = default;

  const TaskProto* task_proto() const { return task_proto_; }

 private:
  const TaskProto* task_proto_;
};

class MemSharedTaskGraph final : public Graph<MemSharedTaskNode, MemSharedTaskEdge> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(MemSharedTaskGraph);
  explicit MemSharedTaskGraph(const Plan& plan);
  ~MemSharedTaskGraph() = default;

  HashSet<int64_t> ComputeLifeTimeSameStreamTaskIds4RegstDescId(
      const RegstDescProto& regst_desc) const;

 private:
  void InitNodes();
  void InitEdges();
  void InitNode2Ancestor();
  bool IsAnyOneReachable(const HashSet<MemSharedTaskNode*>& nodes,
                         const MemSharedTaskNode* ancestor) const;

  const Plan* plan_;
  HashMap<int64_t, MemSharedTaskNode*> task_id2mem_shared_task_node_;
  HashMap<const MemSharedTaskNode*, HashSet<const MemSharedTaskNode*>> node2ancestor_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_GRAPH_MEMORY_SHARED_TASK_GRAPH_H_
