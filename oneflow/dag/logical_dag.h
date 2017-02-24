#ifndef ONEFLOW_DAG_LOGICAL_DAG_H
#define ONEFLOW_DAG_LOGICAL_DAG_H

#include <memory>
#include "dag/dag.h"
#include "layer/base_layer_desc.h"
#include "job/dlnet_conf.pb.h"
#include "job/strategy.pb.h"

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
  
 private:

};

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

class LogicalDag : public Dag {
 public:
  DISALLOW_COPY_AND_MOVE(LogicalDag);
  LogicalDag() = default;
  ~LogicalDag() = default;

  void Init(const std::string& dag_name,
            const DLNetConf& dl_net_conf,
            const Strategy& strategy_conf);

 private:
  void BuildDagStruct(const DLNetConf& dl_net_conf);
  void FillNodeWithParallelConf(const Strategy& strategy_conf);

  LogicalDataNode* NewLogicalDataNode() {
    LogicalDataNode* ret_ptr = new LogicalDataNode;
    ret_ptr->Init();
    RegisterDataNode(std::unique_ptr<LogicalDataNode> (ret_ptr));
    return ret_ptr;
  }

  LogicalOpNode* NewLogicalOpNode() {
    LogicalOpNode* ret_ptr = new LogicalOpNode;
    ret_ptr->Init();
    RegisterOpNode(std::unique_ptr<LogicalOpNode> (ret_ptr));
    return ret_ptr;
  }

};

} // namespace oneflow

#endif // ONEFLOW_DAG_LOGICAL_DAG_H
