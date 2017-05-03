#ifndef ONEFLOW_GRAPH_TASK_GRAPH_H_
#define ONEFLOW_GRAPH_TASK_GRAPH_H_

#include "graph/stage_graph.h"
#include "graph/boxing_task_node.h"
#include "graph/copy_task_node.h"
#include "operator/operator.h"
#include "operator/operator_manager.h"
#include "job/parallel_desc.h"
#include "job/id_manager.h"
#include "job/job_desc.h"

namespace oneflow {

class TaskGraph : public Graph<TaskNode, TaskEdge> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(TaskGraph);
  virtual ~TaskGraph() = default;
  
  // Getters
  const StageGraph* stage_gph() const { return stage_gph_.get(); }
  const ChainGraph* chain_gph() const { return stage_gph_->chain_gph(); }
  const HashMap<CompTaskNode*, CompTaskNode*>& faker2mccoy() {
    return faker2mccoy_;
  }
  std::vector<CompTaskNode*> SortedCompTasksInChain(const ChainNode*) const;

  // Build Exec And Set Produced Registers
  void BuildExecAndEnrollLbn2Regsts();
  void InferShape4LbnInProducedRegsts();
  
  using CompTaskNodeMemFunc = void (CompTaskNode::*)(TaskGraph*);
  virtual CompTaskNodeMemFunc Func4FwBuildExecAndEnrollLbn2Regsts() const = 0;
  virtual CompTaskNodeMemFunc Func4FwInferShape4LbnInProducedRegsts() const = 0;

 protected:
  TaskGraph() = default;
  void BuildFromChainGph(std::unique_ptr<ChainGraph>&& chain_gph,
                         bool need_bp,
                         const std::string& dot_filepath_prefix);
  void EnrollFakerMccoy(CompTaskNode* faker, CompTaskNode* mccoy) {
    CHECK(faker2mccoy_.emplace(faker, mccoy).second);
  }

 private:
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

  void InitCompTaskNodes(Stage2TaskNodesMap* stage2task_nodes);
  void Stage2DeviceCompTaskNodes(const StageNode* stage,
                                 TaskNodesInStage* task_nodes_in_stage);
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
  HashMap<CompTaskNode*, CompTaskNode*> faker2mccoy_;
  

};

} // namespace oneflow

#endif // ONEFLOW_GRAPH_TASK_GRAPH_H_
