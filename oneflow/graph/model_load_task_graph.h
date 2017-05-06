#ifndef ONEFLOW_GRAPH_MODEL_LOAD_TASK_GRAPH_H_
#define ONEFLOW_GRAPH_MODEL_LOAD_TASK_GRAPH_H_

#include "graph/model_update_task_graph.h"
#include "graph/model_load_save_task_graph.h"

namespace oneflow {

class MdLoadTaskGraph final : public MdLoadSaveTaskGraph {
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
      const HashMap<uint64_t, CompTaskNode*>& parallel_id2updt_task,
      ParallelPolicy policy,
      const std::string& dot_path_prefix);

 private:
  void BuildTaskGraph(const ChainNode* update_chain,
                      const std::string& dot_path_prefix);

};

} // namespace oneflow

#endif // ONEFLOW_GRAPH_MODEL_LOAD_TASK_GRAPH_H_
