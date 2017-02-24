#ifndef ONEFLOW_LOGICAL_OP_NODE_H_
#define ONEFLOW_LOGICAL_OP_NODE_H_

#include "dag/op_node.h"
#include "layer/base_layer_desc.h"
#include "job/strategy.pb.h"

namespace oneflow {

class LogicalOpNode : public OpNode {
 public:
  DISALLOW_COPY_AND_MOVE(LogicalOpNode);
  LogicalOpNode() = default;
  ~LogicalOpNode() = default;

  void Init() {
    OpNode::Init();
    // struct style
  }

  const BaseLayerDesc& layer_desc() const {
    return *(layer_desc_.get());
  }
  const ParallelConf& parallel_conf() const {
    return parallel_conf_;
  }

  std::unique_ptr<BaseLayerDesc>& mutable_layer_desc() {
    return layer_desc_;
  }
  ParallelConf& mutable_parallel_conf() {
    return parallel_conf_;
  }

 private:
  std::unique_ptr<BaseLayerDesc> layer_desc_;
  ParallelConf parallel_conf_;

};

} // namespace oneflow

#endif // ONEFLOW_LOGICAL_OP_NODE_H_
