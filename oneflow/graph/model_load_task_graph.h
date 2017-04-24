#ifndef ONEFLOW_GRAPH_MODEL_LOAD_TASK_GRAPH_H_
#define ONEFLOW_GRAPH_MODEL_LOAD_TASK_GRAPH_H_

#include "graph/model_update_task_graph.h"

namespace oneflow {

class MdLoadTaskGraph final : public TaskGraph {
 public:
  OF_DISALLOW_COPY_AND_MOVE(MdLoadTaskGraph);
  MdLoadTaskGraph() = delete;
  ~MdLoadTaskGraph() = default;

  CompTaskNodeMemFunc Func4FwBuildExecAndProducedRegsts() const override {
    return &CompTaskNode::MdLoadFwBuildExecAndProducedRegsts;
  }

  MdLoadTaskGraph(
      const ChainNode* update_chain,
      const std::vector<CompTaskNode*>& sorted_update_tasks);

 private:
  void BuildTaskGraph(const ChainNode* update_chain);
  void InitFaker2Mccoy(
      const std::vector<CompTaskNode*>& sorted_update_tasks);

};

} // namespace oneflow

#endif // ONEFLOW_GRAPH_MODEL_LOAD_TASK_GRAPH_H_
