#ifndef ONEFLOW_DAG_DATA_NODE_H_
#define ONEFLOW_DAG_DATA_NODE_H_

#include "dag/dag_node.h"
#include "dag/data_meta.h"

namespace oneflow {

class DataNode : public DagNode {
 public:
  DISALLOW_COPY_AND_MOVE(DataNode);
  virtual ~DataNode() = default;

  virtual std::shared_ptr<const DataMeta> data_meta() const = 0;
 
 protected:
  DataNode() = default;
  void init() {
    DagNode::init();
  }

 private:

};

} // namespace oneflow

#endif // ONEFLOW_DAG_DATA_NODE_H_
