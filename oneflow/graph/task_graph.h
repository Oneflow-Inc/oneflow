#ifndef ONEFLOW_GRAPH_TASK_GRAPH_H_
#define ONEFLOW_GRAPH_TASK_GRAPH_H_

#include "graph/stage_graph.h"
#include "graph/boxing_task_node.h"
#include "graph/copy_hd_task_node.h"
#include "operator/operator.h"
#include "job/parallel_desc.h"
#include "common/id_manager.h"

namespace oneflow {

class Path;

class TaskGraph final : public Graph<TaskNode, TaskEdge> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(TaskGraph);
  TaskGraph() = delete;
  ~TaskGraph() = default;
  
  TaskGraph(const DLNetConf& dl_net_conf,
            const Strategy& strategy_conf,
            bool need_bp);
  TaskGraph(std::unique_ptr<ChainGraph>&& chain_graph, bool need_bp);

  const StageGraph* stage_graph() const { return stage_graph_.get(); }

 private:
  void BuildFromChainGph(std::unique_ptr<ChainGraph>&& chain_gph, bool need_bp);
  void BuildGraph(bool need_bp);

  template<typename TaskNodeType>
  TaskNodeType* NewTaskNode() {
    static_assert(std::is_base_of<TaskNode, TaskNodeType>::value, "");
    TaskNodeType* ret = new TaskNodeType;
    RegisterNode(ret);
    return ret;
  }

  // Functions about Init
  struct TaskNodesInStage {
    std::vector<TaskNode*> comp_in_task_nodes;
    std::vector<TaskNode*> comp_out_task_nodes;
    BoxingTaskNode* in_boxing_task_node;
    BoxingTaskNode* out_boxing_task_node;
  };
  
  using Stage2TaskNodesMap =
      std::unordered_map<const StageNode*, TaskNodesInStage>;

  void InitCompTaskNodes(Stage2TaskNodesMap* stage2task_nodes);
  void Stage2DeviceCompTaskNodes(const StageNode* stage,
                                 TaskNodesInStage* task_nodes_in_stage,
                                 bool is_first_stage,
                                 bool is_last_stage);
  void Stage2HostCompTaskNodes(const StageNode* stage,
                               TaskNodesInStage* task_nodes_in_stage);
  void InitBoxingTaskNodes(Stage2TaskNodesMap* stage2task_nodes);
  void InitInboxingTaskNode(const StageNode* stage,
                            TaskNodesInStage* task_nodes_in_stage);
  void InitOutBoxingTaskNode(const StageNode* stage,
                             TaskNodesInStage* task_nodes_in_stage);
  void ConnectTaskNodes(const Stage2TaskNodesMap* stage2task_nodes);
  void GenerateRelatedBpNodes(std::vector<TaskNode*> *turning_node_vec);
  void BackwardConnect(const std::vector<TaskNode*>& turning_node_vec);
  void BuildBpStruct();

  std::unique_ptr<const StageGraph> stage_graph_;

};

} // namespace oneflow

#endif // ONEFLOW_GRAPH_TASK_GRAPH_H_
