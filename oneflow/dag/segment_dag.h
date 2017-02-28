#ifndef ONEFLOW_DAG_SEGMENT_DAG_H_
#define ONEFLOW_DAG_SEGMENT_DAG_H_

#include <list>
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

  const std::vector<std::shared_ptr<const BaseLayerDesc>>& layer_desc_vec() const {
    return layer_desc_vec_;
  }
  const ParallelConf& parallel_conf() const {
    return parallel_conf_;
  }
  
  std::vector<std::shared_ptr<const BaseLayerDesc>>& mutable_layer_desc_vec() {
    return layer_desc_vec_;
  }
  ParallelConf& mutable_parallel_conf() {
    return parallel_conf_;
  }

 private:
  std::vector<std::shared_ptr<const BaseLayerDesc>> layer_desc_vec_;
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
  SegmentDataNode* NewSegmentDataNode() {
    SegmentDataNode* ret_ptr = new SegmentDataNode;
    ret_ptr->Init();
    RegisterDataNode(std::unique_ptr<SegmentDataNode> (ret_ptr));
    return ret_ptr;
  }
  SegmentOpNode* NewSegmentOpNode() {
    SegmentOpNode* ret_ptr = new SegmentOpNode;
    ret_ptr->Init();
    RegisterOpNode(std::unique_ptr<SegmentOpNode> (ret_ptr));
    return ret_ptr;
  }

};

} // namespace oneflow

#endif // ONEFLOW_DAG_SEGMENT_DAG_H_
