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

  TaskGraph* task_graph() {
    return task_graph_.get();
  }
  const ChainGraph* chain_graph() const {
    return task_graph_->stage_graph()->chain_graph();
  }

  const std::unordered_map<CompTaskNode*, CompTaskNode*>& faker2mccoy() const {
    return faker2mccoy_;
  }

  typedef void (CompTaskNode::*CompTaskNodeMemFunc)(Path*);
  virtual CompTaskNodeMemFunc MemFunc4FwBuildExecAndProducedRegisters() const;

 protected:
  std::unique_ptr<TaskGraph>& mut_task_graph() { return task_graph_; }
  void AddFakerMccoyPair(CompTaskNode* faker, CompTaskNode* mccoy) {
    CHECK(faker2mccoy_.emplace(faker, mccoy).second);
  }

 private:
  std::unique_ptr<TaskGraph> task_graph_;
  std::unordered_map<CompTaskNode*, CompTaskNode*> faker2mccoy_;

};

} // namespace oneflow

#endif // ONEFLOW_PATH_PATH_H_
