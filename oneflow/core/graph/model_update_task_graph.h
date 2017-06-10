#ifndef ONEFLOW_CORE_GRAPH_MODEL_UPDATE_TASK_GRAPH_H_
#define ONEFLOW_CORE_GRAPH_MODEL_UPDATE_TASK_GRAPH_H_

#include "oneflow/core/graph/task_graph.h"

namespace oneflow {

class MdUpdtTaskGraph final : public TaskGraph {
 public:
  OF_DISALLOW_COPY_AND_MOVE(MdUpdtTaskGraph);
  MdUpdtTaskGraph() = delete;
  ~MdUpdtTaskGraph() = default;

  MdUpdtTaskGraph(const std::string& name,
                  CompTaskNode* diff_acc_task,
                  const std::string& dot_path_prefix);

  CompTaskNode* diff_acc_task() const { return diff_acc_task_; }

 private:
  void BuildTaskGraph(const std::string& dot_path_prefix);

  CompTaskNode* diff_acc_task_;

};

} // namespace oneflow

#endif // ONEFLOW_CORE_GRAPH_MODEL_SAVE_TASK_GRAPH_H_
