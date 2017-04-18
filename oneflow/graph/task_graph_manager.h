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

  void Init(const JobDesc& job_desc);

 private:
  template<typename ValType>
  using ChainAsKeyMap = HashMap<const ChainNode*, ValType>;

  TaskGraphMgr() = default;

  std::unique_ptr<DataTaskGraph> data_task_gph_;

  ChainAsKeyMap<std::unique_ptr<MdUpdtTaskGraph>> md_updt_task_gphs_;
  ChainAsKeyMap<std::unique_ptr<MdLoadTaskGraph>> md_load_task_gphs_;
  ChainAsKeyMap<std::unique_ptr<MdSaveTaskGraph>> md_save_task_gphs_;

};

} // namespace oneflow

#endif // ONEFLOW_GRAPH_TASK_GRAPH_MANAGER_H_
