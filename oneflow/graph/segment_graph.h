#ifndef ONEFLOW_GRAPH_SEGMENT_GRAPH_H_
#define ONEFLOW_GRAPH_SEGMENT_GRAPH_H_

#include <list>
#include "graph/logical_graph.h"

namespace oneflow {

class SegmentNode final : public Node {
 public:
  DISALLOW_COPY_AND_MOVE(SegmentNode);
  SegmentNode() = default;
  ~SegmentNode() = default;

  void Init() {
    Node::Init();
    // struct style
  }

  const std::vector<std::shared_ptr<const Operator>>& op_vec() const {
    return op_vec_;
  }
  std::vector<std::shared_ptr<const Operator>>& mutable_op_vec() {
    return op_vec_;
  }

  const ParallelDesc& parallel_desc() const {
    return *parallel_desc_ptr_;
  }
  std::shared_ptr<const ParallelDesc> parallel_desc_ptr() const {
    return parallel_desc_ptr_;
  }
  std::shared_ptr<const ParallelDesc>& mutable_parallel_desc_ptr() {
    return parallel_desc_ptr_;
  }

 private:
  std::vector<std::shared_ptr<const Operator>> op_vec_;
  std::shared_ptr<const ParallelDesc> parallel_desc_ptr_;

};

class SegmentGraph final : public Graph {
 public:
  DISALLOW_COPY_AND_MOVE(SegmentGraph);
  SegmentGraph() = default;
  ~SegmentGraph() = default;

  void Init(const LogicalGraph* logical_dag);

 private:
  SegmentNode* NewSegmentNode() {
    SegmentNode* ret_ptr = new SegmentNode;
    ret_ptr->Init();
    RegisterNode(ret_ptr);
    return ret_ptr;
  }

};

} // namespace oneflow

#endif // ONEFLOW_GRAPH_SEGMENT_GRAPH_H_
