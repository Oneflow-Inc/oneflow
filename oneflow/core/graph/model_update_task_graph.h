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
                  CompTaskNode* fw_task,
                  CompTaskNode* diff_acc_task);

  CompTaskNode* fw_task() const { return fw_task_; }
  CompTaskNode* diff_acc_task() const { return diff_acc_task_; }
  const char* TypeName() const override { return "MdUpdtTaskGraph"; }

 private:
  void BuildTaskGraph();

  CompTaskNode* fw_task_;
  CompTaskNode* diff_acc_task_;
};

} // namespace oneflow

#endif // ONEFLOW_CORE_GRAPH_MODEL_UPDATE_TASK_GRAPH_H_
