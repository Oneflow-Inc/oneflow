#ifndef ONEFLOW_GRAPH_MODEL_UPDATE_TASK_GRAPH_H_
#define ONEFLOW_GRAPH_MODEL_UPDATE_TASK_GRAPH_H_

#include "graph/task_graph.h"

namespace oneflow {

class MdUpdtTaskGraph final : public TaskGraph {
 public:
  OF_DISALLOW_COPY_AND_MOVE(MdUpdtTaskGraph);
  MdUpdtTaskGraph() = delete;
  ~MdUpdtTaskGraph() = default;

  MdUpdtTaskGraph(
      const ChainNode* data_chain,
      const std::vector<CompTaskNode*>& sorted_bp_comptasks4data_chain);
  
  CompTaskNodeMemFunc Func4FwBuildExecAndEnrollLbn2Regsts() const override {
    return &CompTaskNode::MdUpdtFwBuildExecAndEnrollLbn2Regsts;
  }
  CompTaskNodeMemFunc Func4FwInferShapeOfBlobsInProducedRegsts() const override {
    return &CompTaskNode::MdUpdtFwInferShapeOfBlobsInProducedRegsts;
  }

  CompTaskNode* GetBpTaskFromParallelId(uint64_t parallel_id) const {
    return parallel_id2bp_task_.at(parallel_id);
  }

 private:
   void BuildTaskGraph(const ChainNode* data_chain);

   HashMap<uint64_t, CompTaskNode*> parallel_id2bp_task_;

};

} // namespace oneflow

#endif // ONEFLOW_GRAPH_MODEL_UPDATE_TASK_GRAPH_H_
