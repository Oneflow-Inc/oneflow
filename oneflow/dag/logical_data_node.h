#ifndef ONEFLOW_LOGICAL_DATA_NODE_H_
#define ONEFLOW_LOGICAL_DATA_NODE_H_

#include "dag/data_node.h"
#include "blob/blob_descriptor.h"

namespace oneflow {

class LogicalDataNode : public DataNode {
 public:
  DISALLOW_COPY_AND_MOVE(LogicalDataNode);
  LogicalDataNode() = default;
  ~LogicalDataNode() = default;

  void Init() {
    DataNode::Init();
    // struct style
  }
  
  const BlobDescriptor& blob_desc() const {
    return blob_desc_;
  }
  BlobDescriptor& mutable_blob_desc() {
    return blob_desc_;
  }

 private:
  BlobDescriptor blob_desc_;

};

} // namespace oneflow

#endif // ONEFLOW_LOGICAL_DATA_NODE_H_
