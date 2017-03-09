#ifndef ONEFLOW_GRAPH_TASK_GRAPH_H_
#define ONEFLOW_GRAPH_TASK_GRAPH_H_

#include "graph/stage_graph.h"
#include "graph/task_node.h"
#include "operator/operator.h"
#include "job/parallel_desc.h"
#include "common/id_map.h"

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
  
  void Init(const StageGraph* stage_graph,
            const IDMap& id_map,
            bool need_bp);

 private:
  struct TndsWithinStage {
    std::vector<TaskNode*> compute_in_tnds;
    std::vector<TaskNode*> compute_out_tnds;
    BoxingTnd* in_boxing_tnd;
    BoxingTnd* out_boxing_tnd;
  };
  
  using Stage2TndsMap =
    std::unordered_map<const StageNode*, TndsWithinStage>;
  
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
    TaskNode* ret = node.get();
    RegisterNode(std::move(node));
    return ret;
  }
  
  void InitComputeTnds(const StageGraph* stage_graph,
                       const IDMap& id_map,
                       Stage2TndsMap* stage2tnds);
  void Stage2DeviceComputeTnds(const StageNode* stage,
                               const IDMap& id_map,
                               TndsWithinStage* tnds_within_stage,
                               bool is_first_stage,
                               bool is_last_stage);
  void Stage2HostComputeTnds(const StageNode* stage,
                             const IDMap& id_map,
                             TndsWithinStage* tnds_within_stage);
  void InitBoxingTnds(const StageGraph* stage_graph,
                      const IDMap& id_map,
                      Stage2TndsMap* stage2tnds);
  void InitInboxingTnd(const StageNode* stage,
                       const IDMap& id_map,
                       TndsWithinStage* tnds_within_stage);
  void InitOutBoxingTnd(const StageNode* stage,
                        const IDMap& id_map,
                        TndsWithinStage* tnds_within_stage);
  void ConnectTnds(const StageGraph* stage_graph,
                   const Stage2TndsMap* stage2tnds);
  void GenerateRelatedBpNodes(
      std::function<void(const TaskNode*, TaskNode*)> add_fw_bp_pair,
      const std::unordered_map<const TaskNode*, TaskNode*>& fw_node2bp_node,
      std::vector<TaskNode*> *turning_node_vec);
  void BackwardConnect(
      const std::unordered_map<const TaskNode*, TaskNode*>& fw_node2bp_node,
      const std::unordered_map<TaskNode*, const TaskNode*>& bp_node2fw_node,
      const std::vector<TaskNode*>& turning_node_vec);
  void BuildBpStruct();

};

} // namespace oneflow

#endif // ONEFLOW_GRAPH_TASK_GRAPH_H_
