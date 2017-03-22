#ifndef ONEFLOW_PATH_PATH_BUILDER_H_
#define ONEFLOW_PATH_PATH_BUILDER_H_

#include "common/util.h"
#include "graph/transfm_graph.h"

namespace oneflow {

class PathBuilder {
 public:
  OF_DISALLOW_COPY_AND_MOVE(PathBuilder);
  PathBuilder() = default;
  ~PathBuilder() = default;

 private:
  std::shared_ptr<TaskGraph> task_graph_;
  std::vector<std::unique_ptr<TransfmGraph>> transfm_graph_vec_;

};

} // namespace oneflow

#endif // ONEFLOW_PATH_PATH_BUILDER_H_
