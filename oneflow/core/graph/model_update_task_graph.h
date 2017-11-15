#ifndef ONEFLOW_CORE_GRAPH_MODEL_UPDATE_TASK_GRAPH_H_
#define ONEFLOW_CORE_GRAPH_MODEL_UPDATE_TASK_GRAPH_H_

#include "oneflow/core/common/util.h"
#include "oneflow/core/graph/chain_graph.h"
#include "oneflow/core/graph/model_diff_accumulate_task_graph.h"
#include "oneflow/core/graph/task_graph.h"

namespace oneflow {

class MdUpdtTaskGraph final : public TaskGraph {
 public:
  OF_DISALLOW_COPY_AND_MOVE(MdUpdtTaskGraph);
  MdUpdtTaskGraph() = delete;
  ~MdUpdtTaskGraph() = default;

  MdUpdtTaskGraph(
      const std::string& name, uint32_t random_seed,
      const ChainNode* data_chain,
      const std::vector<CompTaskNode*>& sorted_diff_acc_tasks,
      const std::vector<CompTaskNode*>& sorted_fw_comptasks4data_chain);

  const char* TypeName() const override { return "MdUpdtTaskGraph"; }

 private:
  void BuildTaskGraph(uint32_t random_seed, const ChainNode* data_chain);
  CompTaskNode* GetDiffAccTask(uint64_t parallel_id) {
    auto it = parallel_id2diff_acc_task_.find(parallel_id);
    if (it == parallel_id2diff_acc_task_.end()) { return nullptr; }
    return it->second;
  }
  CompTaskNode* GetFwTask(uint64_t parallel_id) {
    auto it = parallel_id2fw_task_.find(parallel_id);
    CHECK(it != parallel_id2fw_task_.end());
    return it->second;
  }

  const MdDiffAccTaskGraph* diff_acc_gph_;
  HashMap<uint64_t, CompTaskNode*> parallel_id2diff_acc_task_;
  HashMap<uint64_t, CompTaskNode*> parallel_id2fw_task_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_GRAPH_MODEL_UPDATE_TASK_GRAPH_H_
