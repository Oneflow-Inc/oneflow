#ifndef ONEFLOW_GRAPH_TASK_GRAPH_MANAGER_H_
#define ONEFLOW_GRAPH_TASK_GRAPH_MANAGER_H_

#include "job/job_desc.h"
#include "graph/data_task_graph.h"
#include "graph/model_load_task_graph.h"
#include "graph/model_save_task_graph.h"
#include "graph/model_update_task_graph.h"

namespace oneflow {

class TaskGraphMgr {
 public:
  OF_DISALLOW_COPY_AND_MOVE(TaskGraphMgr);
  ~TaskGraphMgr() = default;

  static TaskGraphMgr& Singleton() {
    static TaskGraphMgr obj;
    return obj;
  }

  void BuildGraphs();
  void InferShape4Regsts();
  void AllTaskNodesToProto(PbRpf<TaskProto>*);

 private:
  TaskGraphMgr() = default;

  std::vector<std::unique_ptr<TaskGraph>> ordered_task_gphs_;

};

} // namespace oneflow

#endif // ONEFLOW_GRAPH_TASK_GRAPH_MANAGER_H_
