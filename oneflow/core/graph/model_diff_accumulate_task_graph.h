#ifndef ONEFLOW_CORE_GRAPH_MODEL_DIFF_ACCUMULATE_TASK_GRAPH_H_
#define ONEFLOW_CORE_GRAPH_MODEL_DIFF_ACCUMULATE_TASK_GRAPH_H_

#include "oneflow/core/graph/task_graph.h"

namespace oneflow {

class MdDiffAccTaskGraph final : public TaskGraph {
 public:
  OF_DISALLOW_COPY_AND_MOVE(MdDiffAccTaskGraph);
  MdDiffAccTaskGraph() = delete;
  ~MdDiffAccTaskGraph() = default;

  MdDiffAccTaskGraph(
      const std::string& name,
      const ChainNode* data_chain,
      const std::vector<CompTaskNode*>& sorted_fw_comptasks4data_chain,
      const std::string& dot_path_prefix);
  
  CompTaskNode* GetFwTaskFromParallelId(uint64_t parallel_id) const {
    return parallel_id2fw_task_.at(parallel_id);
  }

 private:
   void BuildTaskGraph(const ChainNode* data_chain,
                       const std::string& dot_path_prefix);

   HashMap<uint64_t, CompTaskNode*> parallel_id2fw_task_;

};

} // namespace oneflow

#endif // ONEFLOW_CORE_GRAPH_MODEL_DIFF_ACCUMULATE_TASK_GRAPH_H_
