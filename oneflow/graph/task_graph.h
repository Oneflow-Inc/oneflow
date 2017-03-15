#ifndef ONEFLOW_GRAPH_TASK_GRAPH_H_
#define ONEFLOW_GRAPH_TASK_GRAPH_H_

#include "graph/stage_graph.h"
#include "graph/task_node.h"
#include "operator/operator.h"
#include "job/parallel_desc.h"
#include "common/id_map.h"
#include "blob/blob_descriptor.h"

namespace oneflow {

class TaskEdge final : public Edge {
 public:
  DISALLOW_COPY_AND_MOVE(TaskEdge);
  TaskEdge() = default;
  ~TaskEdge() = default;
  
  void Init() {
    Edge::Init();
  }

 private:
};

class TaskGraph final : public Graph {
 public:
  DISALLOW_COPY_AND_MOVE(TaskGraph);
  TaskGraph() = default;
  ~TaskGraph() = default;
  
  void Init(std::shared_ptr<const StageGraph> stage_graph,
            const IDMap& id_map,
            bool need_bp);

 private:
  struct TaskNodesWithinStage {
    std::vector<TaskNode*> comp_in_task_nodes;
    std::vector<TaskNode*> comp_out_task_nodes;
    BoxingTaskNode* in_boxing_task_node;
    BoxingTaskNode* out_boxing_task_node;
  };
  
  using Stage2TaskNodesMap =
    std::unordered_map<const StageNode*, TaskNodesWithinStage>;
  
  template<typename TaskNodeType>
  TaskNodeType* NewTaskNode() {
    static_assert(std::is_base_of<TaskNode, TaskNodeType>::value, "");
    TaskNodeType* ret = new TaskNodeType;
    ret->Init();
    RegisterNode(ret);
    return ret;
  }

  TaskEdge* NewTaskEdge() {
    TaskEdge* ret = new TaskEdge;
    ret->Init();
    RegisterEdge(ret);
    return ret;
  }

  TaskNode* ConstructBpNode(TaskNode* fw_node) {
    std::unique_ptr<TaskNode> node = fw_node->CloneWithOnlyTaskProperty();
    node->SetBpNode();
    TaskNode* ret = node.get();
    RegisterNode(std::move(node));
    return ret;
  }
  
  void InitCompTaskNodes(const StageGraph* stage_graph,
                         const IDMap& id_map,
                         Stage2TaskNodesMap* stage2task_nodes);
  void Stage2DeviceCompTaskNodes(const StageNode* stage,
                                 const IDMap& id_map,
                                 TaskNodesWithinStage* task_nodes_within_stage,
                                 bool is_first_stage,
                                 bool is_last_stage);
  void Stage2HostCompTaskNodes(const StageNode* stage,
                               const IDMap& id_map,
                               TaskNodesWithinStage* task_nodes_within_stage);
  void InitBoxingTaskNodes(const StageGraph* stage_graph,
                           const IDMap& id_map,
                           Stage2TaskNodesMap* stage2task_nodes);
  void InitInboxingTaskNode(const StageNode* stage,
                            const IDMap& id_map,
                            TaskNodesWithinStage* task_nodes_within_stage);
  void InitOutBoxingTaskNode(const StageNode* stage,
                             const IDMap& id_map,
                             TaskNodesWithinStage* task_nodes_within_stage);
  void ConnectTaskNodes(const StageGraph* stage_graph,
                        const Stage2TaskNodesMap* stage2task_nodes);
  void GenerateRelatedBpNodes(
      std::function<void(const TaskNode*, TaskNode*)> add_fw_bp_pair,
      const std::unordered_map<const TaskNode*, TaskNode*>& fw_node2bp_node,
      std::vector<TaskNode*> *turning_node_vec);
  void BackwardConnect(
      const std::unordered_map<const TaskNode*, TaskNode*>& fw_node2bp_node,
      const std::unordered_map<TaskNode*, const TaskNode*>& bp_node2fw_node,
      const std::vector<TaskNode*>& turning_node_vec);
  void BuildBpStruct();

  std::shared_ptr<const StageGraph> stage_graph_;

};

} // namespace oneflow

#endif // ONEFLOW_GRAPH_TASK_GRAPH_H_
