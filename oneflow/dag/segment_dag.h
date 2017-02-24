#ifndef ONEFLOW_DAG_SEGMENT_DAG_H_
#define ONEFLOW_DAG_SEGMENT_DAG_H_

#include "dag/logical_dag.h"

namespace oneflow {

class SegmentDataNode final : public DataNode {
 public:
  DISALLOW_COPY_AND_MOVE(SegmentDataNode);
  SegmentDataNode() = default;
  ~SegmentDataNode() = default;

  void Init() {
    DataNode::Init();
  }

 private:
};

class SegmentOpNode final : public OpNode {
 public:
  DISALLOW_COPY_AND_MOVE(SegmentOpNode);
  SegmentOpNode() = default;
  ~SegmentOpNode() = default;

  void Init() {
    OpNode::Init();
  }

 private:
};

class SegmentDag final : public Dag {
 public:
  DISALLOW_COPY_AND_MOVE(SegmentDag);
  SegmentDag() = default;
  ~SegmentDag() = default;

  void Init(const std::string& dag_name, const LogicalDag& logical_dag);

 private:

};

} // namespace oneflow

#endif // ONEFLOW_DAG_SEGMENT_DAG_H_
