#ifndef ONEFLOW_PATH_PATH_H_
#define ONEFLOW_PATH_PATH_H_

#include "graph/build_graph.h"

namespace oneflow {

class Path {
 public:
  OF_DISALLOW_COPY_AND_MOVE(Path);
  Path() = default;
  virtual ~Path() = default;

 protected:
  std::unique_ptr<TaskGraph>& mut_task_graph() { return task_graph_; }

 private:
  std::unique_ptr<TaskGraph> task_graph_;

};

} // namespace oneflow

#endif // ONEFLOW_PATH_PATH_H_
