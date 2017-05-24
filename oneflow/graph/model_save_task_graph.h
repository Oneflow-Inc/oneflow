#ifndef ONEFLOW_GRAPH_MODEL_SAVE_TASK_GRAPH_H_
#define ONEFLOW_GRAPH_MODEL_SAVE_TASK_GRAPH_H_

#include "graph/task_graph.h"

namespace oneflow {

class MdSaveTaskGraph final : public TaskGraph {
 public:
  OF_DISALLOW_COPY_AND_MOVE(MdSaveTaskGraph);
  MdSaveTaskGraph() = delete;
  ~MdSaveTaskGraph() = default;

  MdSaveTaskGraph(const std::string& name,
                  CompTaskNode* update_task,
                  const std::string& dot_path_prefix);

  CompTaskNodeMemFunc Func4FwBuildExecAndEnrollLbn2Regsts() const override {
    return &CompTaskNode::MdSaveFwBuildExecAndEnrollLbn2Regsts;
  }
  CompTaskNodeMemFunc Func4FwInferShapeOfBlobsInProducedRegsts() const override {
    return &CompTaskNode::MdSaveFwInferShapeOfBlobsInProducedRegsts;
  }

  CompTaskNode* update_task() const { return update_task_; }

 private:
  void BuildTaskGraph(const std::string& dot_path_prefix);

  CompTaskNode* update_task_;

};

} // namespace oneflow

#endif // ONEFLOW_GRAPH_MODEL_SAVE_TASK_GRAPH_H_
