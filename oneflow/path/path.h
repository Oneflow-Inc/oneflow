#ifndef ONEFLOW_PATH_PATH_H_
#define ONEFLOW_PATH_PATH_H_

#include "graph/task_graph.h"
#include "operator/operator_factory.h"

namespace oneflow {

class Path {
 public:
  OF_DISALLOW_COPY_AND_MOVE(Path);
  Path() = default;
  virtual ~Path() = default;

  TaskGraph* task_gph() { return task_gph_.get(); }
  const ChainGraph* chain_gph() const;

  virtual CompTaskNode* Faker2Mccoy(CompTaskNode*) const;

  typedef void (CompTaskNode::*CompTaskNodeMemFunc)(Path*);
  virtual CompTaskNodeMemFunc Func4FwBuildExecAndProducedRegisters() const = 0;

  virtual const ChainNode* GetDataChain() const = 0;

 protected:
  std::unique_ptr<TaskGraph>& mut_task_gph() { return task_gph_; }
  void BuildExecAndProducedRegistersAndSubscribeInPath();

 private:
  std::unique_ptr<TaskGraph> task_gph_;

};

inline const ChainGraph* Path::chain_gph() const {
  return task_gph_->stage_gph()->chain_gph();
}

inline void Path::BuildExecAndProducedRegistersAndSubscribeInPath() {
  for (TaskNode& node : *task_gph_) {
    node.BuildExecAndProducedRegistersAndSubscribeInPath(this);
  }
}

} // namespace oneflow

#endif // ONEFLOW_PATH_PATH_H_
