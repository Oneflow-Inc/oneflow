#ifndef ONEFLOW_GRAPH_MODEL_SAVE_TASK_GRAPH_H_
#define ONEFLOW_GRAPH_MODEL_SAVE_TASK_GRAPH_H_

#include "graph/task_graph.h"

namespace oneflow {

class MdSaveTaskGraph final : public TaskGraph {
 public:
  OF_DISALLOW_COPY_AND_MOVE(MdSaveTaskGraph);
  MdSaveTaskGraph() = delete;
  ~MdSaveTaskGraph() = default;

  MdSaveTaskGraph(const ChainNode*) {
    TODO();
  }

  CompTaskNodeMemFunc Func4FwBuildExecAndProducedRegsts() const override {
    return &CompTaskNode::MdSaveFwBuildExecAndProducedRegsts;
  }

 private:
};

} // namespace oneflow

#endif // ONEFLOW_GRAPH_MODEL_SAVE_TASK_GRAPH_H_
