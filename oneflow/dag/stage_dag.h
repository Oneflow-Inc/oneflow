#ifndef ONEFLOW_DAG_STAGE_DAG_H_
#define ONEFLOW_DAG_STAGE_DAG_H_

#include "dag/segment_dag.h"

namespace oneflow {

class StageDataNode final : public DataNode {
 public:
  DISALLOW_COPY_AND_MOVE(StageDataNode);
  StageDataNode() = default;
  ~StageDataNode() = default;

  void Init() {
    DataNode::Init();
  }

 private:
};

class StageOpNode final : public OpNode {
 public:
  DISALLOW_COPY_AND_MOVE(StageOpNode);
  StageOpNode() = default;
  ~StageOpNode() = default;

  void Init() {
    OpNode::Init();
    // struct style
  }

  const std::vector<std::shared_ptr<const BaseLayerDesc>>& layer_desc_vec() const {
    return layer_desc_vec_;
  }
  const ParallelDesc& parallel_desc() const {
    return *parallel_desc_ptr_;
  }
  const std::shared_ptr<const ParallelDesc>& parallel_desc_ptr() const {
    return parallel_desc_ptr_;
  }
  const MachineId& machine_id() const {
    return machine_id_;
  }
  
  std::vector<std::shared_ptr<const BaseLayerDesc>>& mutable_layer_desc_vec() {
    return layer_desc_vec_;
  }
  std::shared_ptr<const ParallelDesc>& mutable_parallel_desc_ptr() {
    return parallel_desc_ptr_;
  }
  MachineId& mutable_machine_id() {
    return machine_id_;
  }

 private:
  std::vector<std::shared_ptr<const BaseLayerDesc>> layer_desc_vec_;
  std::shared_ptr<const ParallelDesc> parallel_desc_ptr_;
  MachineId machine_id_;

};

class StageDag final : public Dag {
 public:
  using OpNodePtrType = StageOpNode*;

  DISALLOW_COPY_AND_MOVE(StageDag);
  StageDag() = default;
  ~StageDag() = default;

  void Init(const std::string& dag_name,
            std::shared_ptr<const SegmentDag> segment_dag);

 private:
  StageDataNode* NewStageDataNode() {
    StageDataNode* ret_ptr = new StageDataNode;
    ret_ptr->Init();
    RegisterDataNode(std::unique_ptr<StageDataNode> (ret_ptr));
    return ret_ptr;
  }
  StageOpNode* NewStageOpNode() {
    StageOpNode* ret_ptr = new StageOpNode;
    ret_ptr->Init();
    RegisterOpNode(std::unique_ptr<StageOpNode> (ret_ptr));
    return ret_ptr;
  }

};

} // namespace oneflow

#endif // ONEFLOW_DAG_STAGE_DAG_H_
