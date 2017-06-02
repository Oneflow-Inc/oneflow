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

#endif // ONEFLOW_GRAPH_MODEL_UPDATE_TASK_GRAPH_H_
