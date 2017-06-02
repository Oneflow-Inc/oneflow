#ifndef ONEFLOW_GRAPH_TASK_GRAPH_H_
#define ONEFLOW_GRAPH_TASK_GRAPH_H_

#include "oneflow/core/graph/stage_graph.h"
#include "oneflow/core/graph/boxing_task_node.h"
#include "oneflow/core/graph/copy_task_node.h"
#include "oneflow/core/operator/operator.h"
#include "oneflow/core/operator/operator_manager.h"
#include "oneflow/core/compile/parallel_desc.h"
#include "oneflow/core/common/id_manager.h"
#include "oneflow/core/common/job_desc.h"

namespace oneflow {

class TaskGraph : public Graph<TaskNode, TaskEdge> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(TaskGraph);
  virtual ~TaskGraph() = default;
  
  // Getters
  const StageGraph* stage_gph() const { return stage_gph_.get(); }
  const ChainGraph* chain_gph() const { return stage_gph_->chain_gph(); }
  std::vector<CompTaskNode*> SortedCompTasksInChain(const ChainNode*) const;

  void InferShapeOfBlobsInProducedRegsts();
  
  const std::string& name() const { return name_; }

 protected:
  TaskGraph() = default;

  template<typename CompTaskNodeType>
  void BuildFromChainGph(std::unique_ptr<ChainGraph>&& chain_gph,
                         bool need_bp,
                         const std::string& dot_filepath_prefix);
  void BuildExecAndEnrollLbn2Regsts();

  std::string& mut_name() { return name_; }

 private:
  template<typename CompTaskNodeType>
  void BuildFromStageGph(bool need_bp,
                         const std::string& dot_filepath_prefix);

  template<typename TaskNodeType>
  TaskNodeType* NewTaskNode() {
    static_assert(std::is_base_of<TaskNode, TaskNodeType>::value, "");
    TaskNodeType* ret = new TaskNodeType;
    EnrollNode(ret);
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
      HashMap<const StageNode*, TaskNodesInStage>;

  template<typename TaskNodeType>
  void InitCompTaskNodes(Stage2TaskNodesMap* stage2task_nodes);
  template<typename TaskNodeType>
  void Stage2DeviceCompTaskNodes(const StageNode* stage,
                                 TaskNodesInStage* task_nodes_in_stage);
  template<typename TaskNodeType>
  void Stage2HostCompTaskNodes(const StageNode* stage,
                               TaskNodesInStage* task_nodes_in_stage);
  void InitBoxingTaskNodes(Stage2TaskNodesMap* stage2task_nodes);
  void InitInboxingTaskNode(const StageNode* stage,
                            TaskNodesInStage* task_nodes_in_stage);
  void InitOutBoxingTaskNode(const StageNode* stage,
                             TaskNodesInStage* task_nodes_in_stage);
  void ConnectBoxingTaskNodes(const Stage2TaskNodesMap* stage2task_nodes);
  void GenerateRelatedBpNodes(std::vector<TaskNode*> *turning_node_vec);
  void BackwardConnect(const std::vector<TaskNode*>& turning_node_vec);
  void BuildBpStruct();

  std::unique_ptr<const StageGraph> stage_gph_;
  std::string name_; 

};

} // namespace oneflow

#endif // ONEFLOW_GRAPH_TASK_GRAPH_H_
