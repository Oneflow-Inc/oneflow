#ifndef ONEFLOW_DAG_LOGICAL_DAG_H
#define ONEFLOW_DAG_LOGICAL_DAG_H

#include <memory>
#include "dag/dag.h"
#include "layer/base_layer_desc.h"
#include "job/dlnet_conf.pb.h"
#include "job/strategy.pb.h"
#include "job/parallel_desc.h"

namespace oneflow {

class LogicalNode : public DagNode {
 public:
  DISALLOW_COPY_AND_MOVE(LogicalNode);
  LogicalNode() = default;
  ~LogicalNode() = default;

  void Init() {
    DagNode::Init();
    // struct style
  }

  const BaseLayerDesc& layer_desc() const {
    return *layer_desc_ptr_;
  }
  std::shared_ptr<const BaseLayerDesc> layer_desc_ptr() const {
    return layer_desc_ptr_;
  }
  std::shared_ptr<const BaseLayerDesc>& mutable_layer_desc_ptr() {
    return layer_desc_ptr_;
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
  std::shared_ptr<const BaseLayerDesc> layer_desc_ptr_;
  std::shared_ptr<const ParallelDesc> parallel_desc_ptr_;

};

class LogicalDag : public Dag {
 public:
  DISALLOW_COPY_AND_MOVE(LogicalDag);
  LogicalDag() = default;
  ~LogicalDag() = default;

  void Init(const DLNetConf& dl_net_conf,
            const Strategy& strategy_conf);

 private:
  void BuildDagStruct(const DLNetConf& dl_net_conf);
  void FillNodeWithParallelDesc(const Strategy& strategy_conf);
  //void ConnectLogicalNodePtr();

  LogicalNode* NewLogicalNode() {
    LogicalNode* ret_ptr = new LogicalNode;
    ret_ptr->Init();
    RegisterNode(ret_ptr);
    return ret_ptr;
  }

};

} // namespace oneflow

#endif // ONEFLOW_DAG_LOGICAL_DAG_H
