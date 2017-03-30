#ifndef ONEFLOW_GRAPH_BOXING_TASK_NODE_H_
#define ONEFLOW_GRAPH_BOXING_TASK_NODE_H_

#include "graph/task_node.h"

namespace oneflow {

class BoxingTaskNode : public TaskNode {
 public:
  OF_DISALLOW_COPY_AND_MOVE(BoxingTaskNode);
  BoxingTaskNode() = default;
  virtual ~BoxingTaskNode() = default;

 protected:
  virtual void InitWithFwNode(TaskNode* fw_node) override {
    TaskNode::InitWithFwNode(fw_node);
  }
 private:
  
};

} // namespace oneflow

#endif // ONEFLOW_GRAPH_BOXING_TASK_NODE_H_
