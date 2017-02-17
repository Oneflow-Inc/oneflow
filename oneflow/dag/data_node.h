#ifndef ONEFLOW_DAG_DATA_NODE_H_
#define ONEFLOW_DAG_DATA_NODE_H_

#include "dag/dag_node.h"
#include "dag/data_meta.h"

namespace oneflow {

class DataNode : public DagNode {
 public:
  DISALLOW_COPY_AND_MOVE(DataNode);

  DataNode() = default;
  ~DataNode() = default;

  void init(std::shared_ptr<DataMeta> data_meta) {
    DagNode::init();
    data_meta_ = data_meta;
  }

  std::shared_ptr<const DataMeta> data_meta() const { 
    return data_meta_;
  }
  std::shared_ptr<DataMeta> mutable_data_meta() { return data_meta_; }

 private:
  std::shared_ptr<DataMeta> data_meta_;
};

} // namespace oneflow

#endif // ONEFLOW_DAG_DATA_NODE_H_
