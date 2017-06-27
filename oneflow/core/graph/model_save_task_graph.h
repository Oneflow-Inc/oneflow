#ifndef ONEFLOW_CORE_GRAPH_MODEL_SAVE_TASK_GRAPH_H_
#define ONEFLOW_CORE_GRAPH_MODEL_SAVE_TASK_GRAPH_H_

#include "oneflow/core/graph/task_graph.h"

namespace oneflow {

class MdSaveTaskGraph final : public TaskGraph {
 public:
  OF_DISALLOW_COPY_AND_MOVE(MdSaveTaskGraph);
  MdSaveTaskGraph() = delete;
  ~MdSaveTaskGraph() = default;

  MdSaveTaskGraph(const std::string& name,
                  CompTaskNode* update_task);

  CompTaskNode* update_task() const { return update_task_; }
  const char* TypeName() const override { return "MdSaveTaskGraph"; }

 private:
  void BuildTaskGraph();

  CompTaskNode* update_task_;
};

} // namespace oneflow

#endif // ONEFLOW_CORE_GRAPH_MODEL_SAVE_TASK_GRAPH_H_
