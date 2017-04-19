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
  
  CompTaskNodeMemFunc Func4FwBuildExecAndProducedRegsts() const override {
    return &CompTaskNode::MdUpdtFwBuildExecAndProducedRegsts;
  }

 private:
   void BuildTaskGraph(const ChainNode* data_chain);
   void InitFaker2MccoyAndParallelId2UpdtMap(
       const std::vector<CompTaskNode*>& sorted_bp_comptasks4data_chain,
       HashMap<uint64_t, CompTaskNode*>* parallel_id2updt);
   void CompleteUpdateTaskAndFwTask(
       const std::vector<CompTaskNode*>& sorted_bp_comptasks4data_chain,
       const HashMap<uint64_t, CompTaskNode*>& parallel_id2updt);
};

} // namespace oneflow

#endif // ONEFLOW_GRAPH_MODEL_UPDATE_TASK_GRAPH_H_
