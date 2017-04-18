#ifndef ONEFLOW_GRAPH_MODEL_SAVE_TASK_GRAPH_H_
#define ONEFLOW_GRAPH_MODEL_SAVE_TASK_GRAPH_H_

#include "graph/task_graph.h"

namespace oneflow {

class MdSaveTaskGraph final : public TaskGraph {
 public:
  OF_DISALLOW_COPY_AND_MOVE(MdSaveTaskGraph);
  MdSaveTaskGraph() = delete;
  ~MdSaveTaskGraph() = default;

  MdSaveTaskGraph(
      const ChainNode* update_chain,
      const std::vector<CompTaskNode*>& sorted_update_tasks);

  CompTaskNodeMemFunc Func4FwBuildExecAndProducedRegsts() const override {
    return &CompTaskNode::MdSaveFwBuildExecAndProducedRegsts;
  }

 private:
    void BuildTaskGraph(const ChainNode* update_chain);
    void InitFaker2Mccoy(
        const std::vector<CompTaskNode*>& sorted_updt_tasks);
};

} // namespace oneflow

#endif // ONEFLOW_GRAPH_MODEL_SAVE_TASK_GRAPH_H_
