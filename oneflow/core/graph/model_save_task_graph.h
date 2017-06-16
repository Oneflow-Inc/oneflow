#ifndef ONEFLOW_CORE_GRAPH_MODEL_SAVE_TASK_GRAPH_H_
#define ONEFLOW_CORE_GRAPH_MODEL_SAVE_TASK_GRAPH_H_

#include "oneflow/core/graph/task_graph.h"
#include "oneflow/core/graph/model_update_comp_task_node.h"

namespace oneflow {

class MdSaveTaskGraph final : public TaskGraph {
 public:
  OF_DISALLOW_COPY_AND_MOVE(MdSaveTaskGraph);
  MdSaveTaskGraph() = delete;
  ~MdSaveTaskGraph() = default;

  MdSaveTaskGraph(const std::string& name,
                  MdUpdtCompTaskNode* update_task,
                  const std::string& dot_path_prefix);

  MdUpdtCompTaskNode* update_task() const { return update_task_; }

 private:
  void BuildTaskGraph(const std::string& dot_path_prefix);

  MdUpdtCompTaskNode* update_task_;

};

} // namespace oneflow

#endif // ONEFLOW_CORE_GRAPH_MODEL_SAVE_TASK_GRAPH_H_
