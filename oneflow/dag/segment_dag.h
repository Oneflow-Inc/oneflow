#ifndef ONEFLOW_DAG_SEGMENT_DAG_H_
#define ONEFLOW_DAG_SEGMENT_DAG_H_

#include <list>
#include "dag/logical_dag.h"

namespace oneflow {

class SegmentNode final : public DagNode {
 public:
  DISALLOW_COPY_AND_MOVE(SegmentNode);
  SegmentNode() = default;
  ~SegmentNode() = default;

  void Init() {
    DagNode::Init();
    // struct style
  }

  const std::vector<std::shared_ptr<const BaseLayerDesc>>& layer_desc_vec() const {
    return layer_desc_vec_;
  }
  std::vector<std::shared_ptr<const BaseLayerDesc>>& mutable_layer_desc_vec() {
    return layer_desc_vec_;
  }

  const ParallelDesc& parallel_desc() const {
    return *parallel_desc_ptr_;
  }
  const std::shared_ptr<const ParallelDesc>& parallel_desc_ptr() const {
    return parallel_desc_ptr_;
  }
  std::shared_ptr<const ParallelDesc>& mutable_parallel_desc_ptr() {
    return parallel_desc_ptr_;
  }

 private:
  std::vector<std::shared_ptr<const BaseLayerDesc>> layer_desc_vec_;
  std::shared_ptr<const ParallelDesc> parallel_desc_ptr_;

};

class SegmentDag final : public Dag {
 public:
  DISALLOW_COPY_AND_MOVE(SegmentDag);
  SegmentDag() = default;
  ~SegmentDag() = default;

  void Init(const LogicalDag* logical_dag);

 private:
  SegmentNode* NewSegmentNode() {
    SegmentNode* ret_ptr = new SegmentNode;
    ret_ptr->Init();
    RegisterNode(ret_ptr);
    return ret_ptr;
  }

};

} // namespace oneflow

#endif // ONEFLOW_DAG_SEGMENT_DAG_H_
