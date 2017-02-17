#ifndef ONEFLOW_DAG_DATA_NODE_H_
#define ONEFLOW_DAG_DATA_NODE_H_

#include "dag/dag_node.h"

namespace oneflow {

class DataNode : public DagNode {
 public:
  DISALLOW_COPY_AND_MOVE(DataNode);
  virtual ~DataNode() = default;
 
 protected:
  DataNode() = default;
  void init() {
    DagNode::init();
  }

 private:

};

} // namespace oneflow

#endif // ONEFLOW_DAG_DATA_NODE_H_
