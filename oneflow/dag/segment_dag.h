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
    // struct style
  }

  const std::vector<std::unique_ptr<BaseLayerDesc>>& layer_desc_vec() const {
    return layer_desc_vec_;
  }
  const ParallelConf& parallel_conf() const {
    return parallel_conf_;
  }
  
  std::vector<std::unique_ptr<BaseLayerDesc>>& mutable_layer_desc_vec() {
    return layer_desc_vec_;
  }
  ParallelConf& mutable_parallel_conf() {
    return parallel_conf_;
  }

 private:
  std::vector<std::unique_ptr<BaseLayerDesc>> layer_desc_vec_;
  ParallelConf parallel_conf_;

};

class SegmentDag final : public Dag {
 public:
  DISALLOW_COPY_AND_MOVE(SegmentDag);
  SegmentDag() = default;
  ~SegmentDag() = default;

  // use shared_ptr to make sure logical_dag is alive
  void Init(const std::string& dag_name,
            std::shared_ptr<const LogicalDag> logical_dag);

 private:

};

} // namespace oneflow

#endif // ONEFLOW_DAG_SEGMENT_DAG_H_
