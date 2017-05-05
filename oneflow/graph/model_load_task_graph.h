#ifndef ONEFLOW_GRAPH_MODEL_LOAD_TASK_GRAPH_H_
#define ONEFLOW_GRAPH_MODEL_LOAD_TASK_GRAPH_H_

#include "graph/model_update_task_graph.h"

namespace oneflow {

class MdLoadTaskGraph final : public TaskGraph {
 public:
  OF_DISALLOW_COPY_AND_MOVE(MdLoadTaskGraph);
  MdLoadTaskGraph() = delete;
  ~MdLoadTaskGraph() = default;

  CompTaskNodeMemFunc Func4FwBuildExecAndEnrollLbn2Regsts() const override {
    return &CompTaskNode::MdLoadFwBuildExecAndEnrollLbn2Regsts;
  }
  CompTaskNodeMemFunc Func4FwInferShapeOfBlobsInProducedRegsts() const override {
    return &CompTaskNode::MdLoadFwInferShapeOfBlobsInProducedRegsts;
  }

  MdLoadTaskGraph(
      const ChainNode* update_chain,
      const std::vector<CompTaskNode*>& sorted_update_tasks,
      ParallelPolicy policy,
      const std::string& dot_path_prefix);

  const HashMap<uint64_t, CompTaskNode*>& parallel_id2updt_task() const {
    return parallel_id2updt_task_;
  }
  ParallelPolicy policy() const { return policy_; }

 private:
  void BuildTaskGraph(const ChainNode* update_chain,
                      const std::string& dot_path_prefix);
  HashMap<uint64_t, CompTaskNode*> parallel_id2updt_task_;
  ParallelPolicy policy_;

};

} // namespace oneflow

#endif // ONEFLOW_GRAPH_MODEL_LOAD_TASK_GRAPH_H_
