#ifndef ONEFLOW_GRAPH_MODEL_LOAD_SAVE_TASK_GRAPH_H_
#define ONEFLOW_GRAPH_MODEL_LOAD_SAVE_TASK_GRAPH_H_

#include "graph/task_graph.h"

namespace oneflow {

class MdLoadSaveTaskGraph : public TaskGraph {
 public:
  OF_DISALLOW_COPY_AND_MOVE(MdLoadSaveTaskGraph);
  virtual ~MdLoadSaveTaskGraph() = default;

  const HashMap<uint64_t, CompTaskNode*>& parallel_id2updt_task() const {
    return parallel_id2updt_task_;
  }
  ParallelPolicy policy() const { return policy_; }

 protected:
  MdLoadSaveTaskGraph() = default;

  HashMap<uint64_t, CompTaskNode*>& mut_parallel_id2updt_task() {
    return parallel_id2updt_task_;
  }
  ParallelPolicy& mut_policy() {
    return policy_;
  }

 private:
  HashMap<uint64_t, CompTaskNode*> parallel_id2updt_task_;
  ParallelPolicy policy_;

};

} // namespace oneflow

#endif // ONEFLOW_GRAPH_MODEL_LOAD_SAVE_TASK_GRAPH_H_
