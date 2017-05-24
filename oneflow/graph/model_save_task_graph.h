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
      const std::string& name,
      const ChainNode* update_chain,
      const HashMap<uint64_t, CompTaskNode*>& parallel_id2updt_task,
      ParallelPolicy data_chain_policy,
      const std::string& dot_path_prefix);

  CompTaskNodeMemFunc Func4FwBuildExecAndEnrollLbn2Regsts() const override {
    return &CompTaskNode::MdSaveFwBuildExecAndEnrollLbn2Regsts;
  }
  CompTaskNodeMemFunc Func4FwInferShapeOfBlobsInProducedRegsts() const override {
    return &CompTaskNode::MdSaveFwInferShapeOfBlobsInProducedRegsts;
  }

  const HashMap<uint64_t, CompTaskNode*>& parallel_id2updt_task() const {
    return parallel_id2updt_task_;
  }

 private:
  void BuildTaskGraph(const ChainNode* update_chain,
                      const std::string& dot_path_prefix);
  
  ParallelPolicy data_chain_policy_;
  HashMap<uint64_t, CompTaskNode*> parallel_id2updt_task_;

};

} // namespace oneflow

#endif // ONEFLOW_GRAPH_MODEL_SAVE_TASK_GRAPH_H_
