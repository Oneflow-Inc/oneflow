#ifndef ONEFLOW_DAG_STAGE_DAG_H_
#define ONEFLOW_DAG_STAGE_DAG_H_

#include "dag/segment_dag.h"

namespace oneflow {

class StageNode final : public DagNode {
 public:
  DISALLOW_COPY_AND_MOVE(StageNode);
  StageNode() = default;
  ~StageNode() = default;

  void Init() {
    DagNode::Init();
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
  DISALLOW_COPY_AND_MOVE(StageDag);
  StageDag() = default;
  ~StageDag() = default;

  void Init(const std::string& dag_name,
            std::shared_ptr<const SegmentDag> segment_dag);

 private:
  StageNode* NewStageNode() {
    StageNode* ret_ptr = new StageNode;
    ret_ptr->Init();
    RegisterNode(ret_ptr);
    return ret_ptr;
  }

};

} // namespace oneflow

#endif // ONEFLOW_DAG_STAGE_DAG_H_
